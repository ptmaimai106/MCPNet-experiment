import importlib
import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import torch.distributed as dist # Removed for single CUDA
import sys
import time
import shutil
import matplotlib.pyplot as plt

from utils import read_args, info_log, cal_cov_component, cal_concept, cal_acc, cal_class_MCP, cal_cov, load_model, check_device, CCD_loss, CKA_loss

# GatherLayer class removed as it's DDP-specific

# =============================================================================
# Get optimizer learning rate
# =============================================================================
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# =============================================================================
# Run one iteration
# =============================================================================
def one_step(model, data, label, loss_funcs, optimizer, args, concept_vectors = None, concept_means = None, class_MCP_dist = None):
    if args.device_id is not None and args.device_id != -1: # Check if device_id is a valid CUDA device
        b_data = data.to(args.device_id)
        b_label = label.to(args.device_id)
    else:
        b_data = data
        b_label = label
        
    optimizer.zero_grad()  
    
    # Model forward  
    l1, l2, l3, l4 = model(b_data)

    # calculate loss
    if l1.shape[0] > 2: # Ensure batch size is sufficient for CKA
        cka_loss = loss_funcs["CKA_loss"]((l1, l2, l3, l4), (1, 2, 3, 4))
    else:
        cka_loss = torch.tensor(0.0, device=args.device_id if args.device_id is not None and args.device_id != -1 else 'cpu')


    ccd_loss = loss_funcs["CCD_loss"]((l1, l2, l3, l4), concept_vectors, concept_means, (1, 2, 3, 4), b_label, class_MCP_dist) * args.CCD_weight

    loss = cka_loss + ccd_loss

    loss.backward()

    optimizer.step()
    
    losses = {
                "CKA_loss" : cka_loss.detach(),
                "CCD_loss" : ccd_loss.detach()
             }
    
    return losses

def test(model, data, label, loss_func, args): # loss_func seems unused here
    if args.device_id is not None and args.device_id != -1:
        b_data = data.to(args.device_id)
        b_label = label.to(args.device_id)
    else:
        b_data = data
        b_label = label

    # Model forward
    l1, l2, l3, l4 = model(b_data)
    # Removed GatherLayer.apply as we are on a single device/process
    # l1, l2, l3, l4, b_label remain as they are from the single process

    losses = {
        # Potentially calculate test losses here if needed, though original code didn't populate it for test
    }
    return losses, l1, l2, l3, l4, b_label # Return b_label as it might be used by caller

# =============================================================================
# Load data, load model (pretrain if needed), define loss function, define optimizer,  
# define learning rate scheduler (if needed), training and validation
# =============================================================================
def runs(args):
    # Load dataset ------------------------------------------------------------
    dataloader = importlib.import_module(args.dataloader)
    dataset, dataset_sizes, all_image_datasets = dataloader.load_data(args) # Assuming load_data handles single GPU setup
    # -------------------------------------------------------------------------
    
    # Define tensorboard for recording ----------------------------------------
    # global_rank will be 0 for single GPU if GPU is present, or -1 if CPU only by original logic
    if args.global_rank in [-1, 0]: 
        with open('{}/logging.txt'.format(args.dst), "a") as f:
            print('Index : {}'.format(args.index), file = f)
            print("dataset : {}".format(args.dataset_name), file = f)
        writer = SummaryWriter('./logs/{}/{}_{}'.format(args.index, args.model.lower(), args.basic_model.lower()))
    # -------------------------------------------------------------------------
    
    start_epoch = 1
    if args.resume:
        resume_data = torch.load(args.weight_path, map_location=args.device_id if args.device_id is not None and args.device_id != -1 else 'cpu')
        args.concept_cha = resume_data['concept_cha']
        start_epoch = resume_data["Epoch"] + 1

    # Load model (load pretrain if needed) ------------------------------------
    model = load_model(args) # load_model should handle moving model to args.device_id
    # -------------------------------------------------------------------------
    
    # Define loss -------------------------------------------------------------
    loss_funcs = {}
    loss_funcs["CCD_loss"] = CCD_loss(args.concept_cha, args.margin) # Assuming CCD_loss is device agnostic or handles it internally
    loss_funcs["CKA_loss"] = CKA_loss(args.concept_cha) # Assuming CKA_loss is device agnostic or handles it internally
    if args.global_rank in [0, -1]:
        print(loss_funcs)
    assert len(loss_funcs) != 0, "Miss define loss"
    # -------------------------------------------------------------------------
    
    # Define optimizer --------------------------------------------------------
    train_optimizer = None
    if args.optimizer == "adam":
        train_optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    if args.optimizer == "sgd":
        train_optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = 0.9)
    if args.optimizer == "adamw":
        train_optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    assert train_optimizer is not None, "Miss define optimizer"
    # -------------------------------------------------------------------------
    
    # Define learning rate scheduler ------------------------------------------
    lr_scheduler = None # Initialize
    if "lr_scheduler" in args and args.lr_scheduler is not None: # check if lr_scheduler is defined
        lr_scheduler = torch.optim.lr_scheduler.StepLR(train_optimizer, step_size = args.lr_scheduler, gamma = 0.1)
    # -------------------------------------------------------------------------
    
    # Define Meters -------------------------------------------------------
    max_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
    last_acc = {'train' : AverageMeter(), 'val' : AverageMeter()}
    # ---------------------------------------------------------------------
    
    # Train and Validation ---------------------------------------------------------------
    concept_vectors = [[], [], [], []]
    concept_means = [[], [], [],[]]
    first_concept_vectors = [[], [], [], []]
    first_concept_means = [[], [], [], []]
    
    train_transform = dataset["train"].dataset.transform
    val_transform = dataset["val"].dataset.transform

    # Determine device for tensors based on args.device_id
    tensor_device = args.device_id if args.device_id is not None and args.device_id != -1 else 'cpu'

    for epoch in range(start_epoch, args.epoch + 1):
        
        if args.global_rank in [-1, 0]:
            info_log('-' * 15, args.global_rank, args.log_type, args.log)
            info_log('Epoch {}/{}'.format(epoch, args.epoch), args.global_rank, args.log_type, args.log)
            
        cov_xxs = [torch.zeros(args.concept_per_layer[0], args.concept_cha[0], args.concept_cha[0], dtype = torch.float64, device=tensor_device), 
                   torch.zeros(args.concept_per_layer[1], args.concept_cha[1], args.concept_cha[1], dtype = torch.float64, device=tensor_device), 
                   torch.zeros(args.concept_per_layer[2], args.concept_cha[2], args.concept_cha[2], dtype = torch.float64, device=tensor_device), 
                   torch.zeros(args.concept_per_layer[3], args.concept_cha[3], args.concept_cha[3], dtype = torch.float64, device=tensor_device)]
        cov_means = [torch.zeros(args.concept_per_layer[0], args.concept_cha[0], 1, dtype = torch.float64, device=tensor_device),  
                     torch.zeros(args.concept_per_layer[1], args.concept_cha[1], 1, dtype = torch.float64, device=tensor_device),  
                     torch.zeros(args.concept_per_layer[2], args.concept_cha[2], 1, dtype = torch.float64, device=tensor_device),  
                     torch.zeros(args.concept_per_layer[3], args.concept_cha[3], 1, dtype = torch.float64, device=tensor_device)]
        Sum_As = [torch.zeros(args.concept_per_layer[0], dtype = torch.float64, device=tensor_device),  
                  torch.zeros(args.concept_per_layer[1], dtype = torch.float64, device=tensor_device),  
                  torch.zeros(args.concept_per_layer[2], dtype = torch.float64, device=tensor_device),  
                  torch.zeros(args.concept_per_layer[3], dtype = torch.float64, device=tensor_device)]
        Square_Sum_As = [torch.zeros(args.concept_per_layer[0], dtype = torch.float64, device=tensor_device),  
                         torch.zeros(args.concept_per_layer[1], dtype = torch.float64, device=tensor_device),  
                         torch.zeros(args.concept_per_layer[2], dtype = torch.float64, device=tensor_device),  
                         torch.zeros(args.concept_per_layer[3], dtype = torch.float64, device=tensor_device)]
        
        # Inference one time to get the concept ====================================================
        if epoch == 1:
            if args.global_rank in [-1, 0]:
                print("First epoch: Extract the concept vectors and concept means!!")
            dataset["train"].dataset.transform = val_transform
            model.train(False) # or model.eval()
            with torch.no_grad():
                nb = len(dataset["train"])
                pbar = enumerate(dataset["train"])
                if args.global_rank in [-1, 0]:  
                    pbar = tqdm.tqdm(pbar, total = nb)  # progress bar

                for step, (data, label) in pbar:
                    losses_test, l1, l2, l3, l4, _ = test(model, data, label, loss_funcs, args) # Get b_label from test
                    features = [l1, l2, l3, l4]
                    Sum_As, Square_Sum_As, cov_xxs, cov_means = cal_cov_component(features, Sum_As, Square_Sum_As, cov_xxs, cov_means, args)
                    
                # Removed dist.all_reduce for Sum_As, Square_Sum_As, etc.
                
                covs = []
                for i in range(len(features)):
                    cov, cov_mean = cal_cov(cov_xxs[i], cov_means[i], Sum_As[i])
                    covs.append(cov)
                    concept_means[i] = cov_mean
                    concept_vectors[i], concept_means[i] = cal_concept(cov, cov_mean)
                    # Ensure tensors are on the correct device if cal_concept doesn't handle it
                    concept_vectors[i] = concept_vectors[i].to(tensor_device)
                    concept_means[i] = concept_means[i].to(tensor_device)

                    first_concept_vectors[i] = concept_vectors[i].type(torch.float32).clone()
                    first_concept_means[i] = concept_means[i].type(torch.float32).clone()
                torch.cuda.empty_cache()

                class_MCP = cal_class_MCP(model, concept_vectors, concept_means, dataset["train"], args.category, args)
            print("Finish extract concept and MCP distribution!!")
        torch.cuda.empty_cache()

        # train phase =================================================================================================
        dataset["train"].dataset.transform = train_transform
        model.train(True)
        # Removed sampler.set_epoch() as it's for DistributedSampler
        
        loss_t = AverageMeter()
        loss_detail_t = {}
        nb = len(dataset["train"])
        pbar = enumerate(dataset["train"])
        if args.global_rank in [-1, 0]:
            pbar = tqdm.tqdm(pbar, total=nb)  # progress bar

        for step, (data, label) in pbar:
            losses = one_step(model = model,  
                                data = data,
                                label = label,  
                                loss_funcs = loss_funcs,  
                                optimizer = train_optimizer,  
                                args = args,  
                                concept_vectors = concept_vectors,  
                                concept_means = concept_means,
                                class_MCP_dist = class_MCP)
            # record losses
            loss_sum_val = 0 # Renamed from 'loss' to avoid conflict with outer scope if any
            for key in losses.keys():
                loss_i = losses[key]
                # Removed dist.all_reduce and division by world_size
                loss_sum_val += loss_i 
                if key not in loss_detail_t.keys():
                    loss_detail_t[key] = AverageMeter()

                if args.global_rank in [-1, 0]:  
                    loss_detail_t[key].update(loss_i.item(), data.size(0)) # Use data.size(0)
                
                losses[key] = losses[key].detach().item() # Keep this for pbar
            if args.global_rank in [-1, 0]:
                loss_t.update(loss_sum_val.item(), data.size(0)) # Use data.size(0)
                pbar.set_postfix(losses)

        if args.global_rank in [-1, 0]:
            writer.add_scalar('Loss/train', loss_t.avg, epoch)
            for key in loss_detail_t.keys():
                writer.add_scalar('{}/train'.format(key), loss_detail_t[key].avg, epoch)
        
        if epoch == 1: # Resetting for the next phase if necessary (logic from original)
            for layer_i in range(len(Sum_As)):
                Sum_As[layer_i].zero_()
                Square_Sum_As[layer_i].zero_()
                cov_xxs[layer_i].zero_()
                cov_means[layer_i].zero_()
        torch.cuda.empty_cache()

        # validation =============================================================================================================    
        # model.train(False) # Set model to eval mode
        model.eval() 

        for phase in ["train", "val"]: # Original code evaluates on "train" set as well in eval mode
            if phase == "train":
                dataset["train"].dataset.transform = val_transform # Use validation transform for 'train' phase eval
            
            correct_t = AverageMeter()
            correct_t5 = AverageMeter()
            loss_val_phase = AverageMeter() # Renamed from loss_t
            loss_detail_val = {} # Renamed from loss_detail_t
            
            with torch.no_grad():
                # total_correct = 0 # Not used elsewhere
                # total_count = 0 # Not used elsewhere
                nb = len(dataset[phase])
                pbar = enumerate(dataset[phase])
                if args.global_rank in [-1, 0]:  
                    pbar = tqdm.tqdm(pbar, total = nb)  # progress bar

                current_Sum_As = [s.clone() for s in Sum_As]
                current_Square_Sum_As = [s.clone() for s in Square_Sum_As]
                current_cov_xxs = [c.clone() for c in cov_xxs]
                current_cov_means = [c.clone() for c in cov_means]

                for step, (data, label) in pbar:
                    # b_label moved to be returned by test function
                    losses_val, l1, l2, l3, l4, b_label_from_test = test(model, data, label, loss_funcs, args)
                    features = [l1, l2, l3, l4]

                    if phase == "train": # Recalculating concepts based on 'train' data in eval mode
                        current_Sum_As, current_Square_Sum_As, current_cov_xxs, current_cov_means = cal_cov_component(
                            features, current_Sum_As, current_Square_Sum_As, current_cov_xxs, current_cov_means, args
                        )
                    else: # phase == "val"
                        resp_top1, resp_top5 = cal_acc(features, class_MCP, concept_vectors, concept_means, args)

                    loss_val_sum = 0
                    for key in losses_val.keys(): # losses_val is from test() which is empty in original
                        loss_i = losses_val[key]
                        # dist.reduce removed
                        # loss_i = loss_i / args.world_size # Removed
                        loss_val_sum += loss_i
                        if key not in loss_detail_val.keys():
                            loss_detail_val[key] = AverageMeter()
                        if args.global_rank in [-1, 0]:  
                            loss_detail_val[key].update(loss_i.item(), data.size(0))
                            
                    if args.global_rank in [-1, 0]:  
                        loss_val_phase.update(loss_val_sum.item() if isinstance(loss_val_sum, torch.Tensor) else loss_val_sum, data.size(0))

                    if phase == "val":
                        # b_label_all and dist.all_gather removed
                        # b_label from the current batch (b_label_from_test) is used directly
                        correct_1 = (resp_top1 == b_label_from_test.unsqueeze(1)).sum().item()
                        correct_5 = (resp_top5 == b_label_from_test.unsqueeze(1)).sum().item()
                        # total_correct += correct_1 # Not used
                        # total_count += b_label_from_test.shape[0] # Not used
                        
                        assert correct_5 >= correct_1, "Error on calculate accuracy"
                        
                        if args.global_rank in [-1, 0]:  
                            correct_t.update(correct_1 / b_label_from_test.shape[0], b_label_from_test.shape[0])
                            correct_t5.update(correct_5 / b_label_from_test.shape[0], b_label_from_test.shape[0])
                
                if phase == "train": # If concepts were recalculated for 'train' set in eval mode
                    # dist.all_reduce removed
                    covs = []
                    # sim_vecs = [] # Not used
                    # sim_means = [] # Not used
                    for i in range(4):
                        cov, cov_mean = cal_cov(current_cov_xxs[i], current_cov_means[i], current_Sum_As[i])
                        covs.append(cov)
                        concept_means[i] = cov_mean # Update global concept_means
                        concept_vectors[i], concept_means[i] = cal_concept(cov, cov_mean) # Update global concept_vectors
                        concept_vectors[i] = concept_vectors[i].to(tensor_device)
                        concept_means[i] = concept_means[i].to(tensor_device)


                    class_MCP = cal_class_MCP(model, concept_vectors, concept_means, dataset["train"], args.category, args)

            if args.global_rank in [-1, 0]:  
                if phase == "val": # Only log validation loss from test, train loss is logged during training phase
                    writer.add_scalar('Loss/{}'.format(phase), loss_val_phase.avg, epoch)
                    for key in losses_val.keys(): # losses_val from test() might be empty
                        writer.add_scalar('{}/{}'.format(key, phase), loss_detail_val[key].avg, epoch)

                writer.add_scalar('Accuracy resp top1/{}'.format(phase), correct_t.avg, epoch)
                writer.add_scalar('Accuracy resp top5/{}'.format(phase), correct_t5.avg, epoch)
                
                if max_acc[phase].avg <= correct_t.avg:
                    last_acc[phase].avg = max_acc[phase].avg # Keep previous best if using .avg directly
                    max_acc[phase].avg = correct_t.avg    # Update current best if using .avg directly
                    
                    if phase == 'val':
                        ACCMeters = correct_t # This captures the current epoch's val acc
                        LOSSMeters = loss_val_phase # This captures current epoch's val loss
                        info_log('save', args.global_rank, args.log_type, args.log)

                        optimizers_state_dict= train_optimizer.state_dict()
                        lr_state_dict = lr_scheduler.state_dict() if lr_scheduler else None
                            
                        save_data = {"Model" : model.state_dict(),
                                     "Epoch" : epoch,
                                     "Optimizer" : optimizers_state_dict,
                                     "lr_scheduler" : lr_state_dict,
                                     "Best ACC" : max_acc[phase].avg, # Save the best acc so far
                                     "concept_cha" : args.concept_cha}
                        torch.save(save_data, f"{args.dst}/best_model.pkl")
                        MCP_data = {"cent_MCP" : class_MCP, # This class_MCP is from the latest train phase re-calculation
                                    "concept_covs" : covs if 'covs' in locals() else None, # covs is from train phase re-calc
                                    "concept_means" : concept_means} # concept_means is from train phase re-calc
                        torch.save(MCP_data, f"{args.dst}/MCP_data.pkl")

                optimizers_state_dict= train_optimizer.state_dict()
                lr_state_dict = lr_scheduler.state_dict() if lr_scheduler else None
                save_data = {"Model" : model.state_dict(),
                             "Epoch" : epoch,
                             "Optimizer" : optimizers_state_dict,
                             "Lr_scheduler" : lr_state_dict,
                             "Current ACC {} phase".format(phase) : correct_t.avg, # More descriptive
                             "concept_cha" : args.concept_cha}
                # Corrected save path for last_model.pkl
                last_model_dir = './pkl/{}/{}_{}'.format(args.index, args.model.lower(), args.basic_model.lower())
                os.makedirs(last_model_dir, exist_ok=True)
                torch.save(save_data, os.path.join(last_model_dir, 'last_model.pkl'))
                
                info_log('Index : {}'.format(args.index), args.global_rank, args.log_type, args.log)
                info_log("dataset : {}".format(args.dataset_name), args.global_rank, args.log_type, args.log)
                info_log("Model name : {}_{}".format(args.model, args.basic_model), args.global_rank, args.log_type, args.log)
                info_log("{} set loss : {:.6f}".format(phase, loss_val_phase.avg), args.global_rank, args.log_type, args.log)
                for key in loss_detail_val.keys(): # This might be empty if losses_val is empty
                    info_log("    {} set {} : {:.6f}".format(phase, key, loss_detail_val[key].avg), args.global_rank, args.log_type, args.log)
                info_log("{} set resp top-1 acc : {:.6f}%".format(phase, correct_t.avg * 100.), args.global_rank, args.log_type, args.log)
                info_log("{} set resp top-5 acc : {:.6f}%".format(phase, correct_t5.avg * 100.), args.global_rank, args.log_type, args.log)
                info_log("{} resp last update : {:.6f}%".format(phase, (correct_t.avg - last_acc[phase].avg) * 100.), args.global_rank, args.log_type, args.log) # Simpler update diff
                info_log("{} set resp max acc : {:.6f}%".format(phase, max_acc[phase].avg * 100.), args.global_rank, args.log_type, args.log)
                info_log("-" * 10, args.global_rank, args.log_type, args.log)
        if lr_scheduler:
            lr_scheduler.step()
    # ---------------------------------------------------------------------

    if args.global_rank in [-1, 0] and 'ACCMeters' in locals() and 'LOSSMeters' in locals(): # Check if defined
        info_log("Best val acc : {:.6f} loss : {:.6f}".format(ACCMeters.avg, LOSSMeters.avg), args.global_rank, args.log_type, args.log)
    elif args.global_rank in [-1, 0]:
        info_log("Training finished. Validation phase might not have found a better model if ACCMeters is not defined.", args.global_rank, args.log_type, args.log)


# =============================================================================
# Templet for recording values
# =============================================================================
class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        self.sum += value * batch
        self.count += batch
        if self.count > 0: # Avoid division by zero
            self.avg = self.sum / self.count
        else:
            self.avg = 0
        
if __name__ == '__main__':
    args = read_args()
    # Set DDP variables - simplified for single CUDA
    args.world_size = 1 # For single GPU, world_size is 1
    args.global_rank = 0 # Main process, or -1 if no CUDA
    args.local_rank = -1 # Not used in single GPU, conventionally -1 or 0

    print("CUDA available:", torch.cuda.is_available())
 

    args.device_id = check_device(args.device, args.train_batch_size, args.val_batch_size)

    if args.device_id.type == 'cuda':
        torch.cuda.set_device(args.device_id)
        args.global_rank = 0
        print(f"Using device: {args.device_id}")
    else:
        args.global_rank = -1
        print("Using CPU")

    # check if it can run on gpu
    # check_device likely returns the device string or ID
    # Let's assume check_device returns a torch.device object or a cuda string like 'cuda:0' or -1 for CPU
    # device_obj = check_device(args.device, args.train_batch_size, args.val_batch_size)
    
    # if isinstance(device_obj, str) and "cuda" in device_obj: # e.g. 'cuda:0'
    #     args.device_id = torch.device(device_obj)
    #     args.global_rank = 0 # GPU is available
    # elif isinstance(device_obj, torch.device) and device_obj.type == "cuda":
    #     args.device_id = device_obj
    #     args.global_rank = 0 # GPU is available
    # elif isinstance(device_obj, int) and device_obj != -1: # e.g. 0 for cuda:0
    #     args.device_id = torch.device(f'cuda:{device_obj}')
    #     args.global_rank = 0
    # else: # CPU
    #     args.device_id = torch.device('cpu')
    #     args.global_rank = -1 # No GPU or user specified CPU

    # if args.device_id.type == 'cuda':
    #     torch.cuda.set_device(args.device_id)
    #     print(f"Using device: {args.device_id}")
    # else:
    #     print("Using CPU")
    #     args.global_rank = -1 # Ensure global_rank is -1 for CPU case

    # args.device_id = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Batch sizes are now direct, no division by world_size
    args.train_total_batch_size = args.train_batch_size
    args.val_total_batch_size = args.val_batch_size
    
    # Removed DDP init_process_group and batch size assert

    args.dst = f"{args.saved_dir}/pkl/{args.index}/{args.model.lower()}_{args.basic_model.lower()}"
    args.log = '{}/logging.txt'.format(args.dst)
    
    # Logging and directory setup (mainly for rank 0, which is our only process now)
    if args.global_rank in [0, -1]: # Should be 0 if GPU, -1 if CPU
        first_time = False
        if not os.path.exists(args.dst):
            first_time = True
            os.makedirs(args.dst, exist_ok=True) # Added exist_ok=True
        
        print(f"Args : {args}")
        if not args.resume and not first_time:
            # Ensure dst is created for the prompt to make sense if it didn't exist but resume is false
            if not os.path.exists(args.dst): os.makedirs(args.dst, exist_ok=True)

            response = input(f"The experiment directory {args.dst} might contain previous results. Are you sure you want to replace it? (y/n)").lower()
            while response != 'y' and response != 'n':
                response = input(f"The experiment directory {args.dst} might contain previous results. Are you sure you want to replace it? (y/n)").lower()
            if response == 'n':
                sys.exit("User chose not to overwrite existing experiment.")

        with open(args.log, "w") as f: # "w" to overwrite log for a new run unless resuming
            print(f"Args : {args}", file = f)
        
        print("Save file to ", args.dst)
        # Ensure source files exist before copying
        files_to_copy = [
            (__file__, __file__), # (source_name_in_script, actual_filename)
            (f"{args.model}.py", f"{args.model}.py"),
        ]
        if args.basic_model == "resnet50":
            files_to_copy.append(("ResNet.py", "ResNet.py"))
        elif args.basic_model == "inceptionv3":
            files_to_copy.append(("inception_net.py", "inception_net.py"))
        
        files_to_copy.extend([
            ("utils/arg_reader.py", "utils/arg_reader.py"),
            ("utils/loss.py", "utils/loss.py")
        ])

        for src_script_name, actual_filename in files_to_copy:
            src_path = os.path.join(os.getcwd(), actual_filename)
            if os.path.exists(src_path):
                shutil.copy(src=src_path, dst=args.dst)
            else:
                print(f"Warning: Source file {src_path} not found, not copied.")
                
        start = time.time()
    
    # args.device_id is already set
    runs(args)
    
    if args.global_rank in [0, -1]: # Should be 0 if GPU, -1 if CPU
        info_log("Train for {:.1f} hours".format((time.time() - start) / 3600), args.global_rank, args.log_type, args.log)