import torch
import tqdm
# import torch.distributed as dist
from typing import Tuple
import importlib

def get_con_num_cha_per_con_num(m, m_is_concept_num, total_cha):
    if m_is_concept_num:
        concept_num = m
        cha_per_con = total_cha // m
    else:
        concept_num = total_cha // m
        cha_per_con = m
    return concept_num, cha_per_con

def load_concept(concept_covs, concept_means, eigen_topk = 1, concept_mode = "pca") -> Tuple[list, list]:
    # Calculate basic correlation
    concept_vecs = []
    concept_means_norm = []
    for layer_i in range(len(concept_means)):
        concept_means_norm.append(concept_means[layer_i] / (torch.norm(concept_means[layer_i], dim = 1, p = 2, keepdim = True) + 1e-16))
    
    if concept_mode.lower() == "pca":
        # PCA
        cur_concept = torch.linalg.eigh(concept_covs[0])[1][:, :, -eigen_topk]
        cur_concept = cur_concept / (torch.norm(cur_concept, dim = 1, keepdim = True, p = 2) + 1e-16)
        mask = torch.sum(concept_means_norm[0] * cur_concept, dim = 1)
        mask = torch.where(mask > 0, 1., -1.)
        cur_concept = cur_concept * mask[:, None]
        cur_concept = cur_concept / (torch.norm(cur_concept, dim = 1, keepdim = True, p = 2) + 1e-16)
        for layer_i in range(1, len(concept_covs)):
            next_concept = torch.linalg.eigh(concept_covs[layer_i])[1][:, :, -eigen_topk]
            next_concept = next_concept / (torch.norm(next_concept, dim = 1, keepdim = True, p = 2) + 1e-16)

            mask = torch.sum(concept_means_norm[layer_i] * next_concept, dim = 1)
            mask = torch.where(mask > 0, 1., -1.)
            next_concept = next_concept * mask[:, None]
            next_concept = next_concept / (torch.norm(next_concept, dim = 1, keepdim = True, p = 2) + 1e-16)
            concept_vecs.append(cur_concept)
            cur_concept = next_concept
        concept_vecs.append(next_concept)
        
    elif concept_mode.lower() == "sevec":
        cur_concept = concept_means_norm[0][..., 0]
        for layer_i in range(1, len(concept_means_norm)):
            next_concept = concept_means_norm[layer_i][..., 0]
            concept_vecs.append(cur_concept)
            cross_layer_sim = abs(cur_concept @ next_concept.T)
            cur_concept = next_concept
        concept_vecs.append(cur_concept)
    else:
        assert False, "No exists cocnept method !"
    
    return concept_vecs, concept_means

def info_log(message, rank = -1, type = ["std"], file = None):
    if rank in [-1, 0]:
        if "std" in type:
            print(message)
        if "log" in type:
            with open(file, "a") as f:
                print(message, file = f)

def cal_cov_component(features, Sum_A, Square_Sum_A, cov_xx, cov_mean, args):
    for layer_i, feat in enumerate(features):
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C // args.concept_cha[layer_i], args.concept_cha[layer_i], H, W).permute(1, 2, 0, 3, 4)
        
        feat = torch.flatten(feat, 2)
        strength = torch.norm(feat, p = 2, dim = 1, keepdim = True)
        ori_feat = feat
        feat = feat * strength
        Sum_A[layer_i] += torch.sum(strength.squeeze(1), dim = 1)
        Square_Sum_A[layer_i] += torch.sum(strength.squeeze(1) ** 2, dim = 1)
        cov_xx[layer_i] += torch.bmm(feat, ori_feat.permute(0, 2, 1))
        cov_mean[layer_i] += torch.sum(feat, dim = -1, keepdim = True)
    return Sum_A, Square_Sum_A, cov_xx, cov_mean

# def cal_cov_component(features, Sum_As, Square_Sum_As, cov_xxs, cov_means, args):
#     """
#     Calculate covariance components for concept extraction
#     """
#     for i, feature in enumerate(features):
#         feature = feature.double()  # Convert to double precision

#         # Ensure feature is on the correct device
#         device = Sum_As[i].device
#         feature = feature.to(device)

#         # Calculate components for each concept
#         for j in range(args.concept_per_layer[i]):
#             # Original code might have DDP-specific operations here
#             # We'll ensure everything stays on the same device

#             # Calculate mean component
#             mean_j = torch.mean(feature, dim=0).view(-1, 1)
#             cov_means[i][j] += mean_j

#             # Calculate covariance component
#             centered_feature = feature - mean_j.view(1, -1)
#             cov_j = torch.matmul(centered_feature.t(), centered_feature)
#             cov_xxs[i][j] += cov_j

#             # Update sum statistics
#             Sum_As[i][j] += feature.shape[0]
#             Square_Sum_As[i][j] += feature.shape[0] ** 2

#     return Sum_As, Square_Sum_As, cov_xxs, cov_means

def cal_cov(cov_xx, cov_mean, Sum_A):
    cov_xx /= Sum_A[:, None, None]
    cov_mean /= Sum_A[:, None, None]
    cov = cov_xx - torch.bmm(cov_mean, cov_mean.permute(0, 2, 1)) #  * (1 / (1. - (Square_Sum_A / (Sum_A ** 2))))[:, None, None]
    return cov, cov_mean[..., 0]

# def cal_cov(cov_xx, cov_mean, Sum_A):
#     """
#     Calculate covariance matrix from components
#     """
#     # Ensure all tensors are on the same device
#     device = cov_xx.device

#     covs = []
#     cov_means = []

#     for j in range(cov_xx.shape[0]):
#         # Original code might have DDP-specific operations
#         # We'll ensure everything stays on the same device

#         # Calculate mean
#         mean_j = cov_mean[j] / Sum_A[j]
#         cov_means.append(mean_j)

#         # Calculate covariance
#         cov_j = cov_xx[j] / Sum_A[j]
#         covs.append(cov_j)

#     return torch.stack(covs), torch.stack(cov_means)



def cal_concept(cov, cov_mean):
    evalue, evector = torch.linalg.eigh(cov)
    cov_mean_norm = cov_mean / (torch.norm(cov_mean, dim = 1, p = 2, keepdim = True) + 1e-16)
    evector[:, :, -1] /= (torch.norm(evector[:, :, -1], dim = 1, p = 2, keepdim = True) + 1e-16)
    # if args.global_rank in [-1, 0]:
    #     print(torch.sum(evector[:, :, -1] * cov_mean_norm[i][..., 0], dim = 1))
    mask = torch.where(torch.sum(evector[:, :, -1] * cov_mean_norm, dim = 1) > 0, 1, -1)
    concept_vector = evector[:, :, -1] * mask[:, None]
    concept_vector /= (torch.norm(concept_vector, dim = 1, p = 2, keepdim = True) + 1e-16)
    concept_vector = concept_vector.type(torch.float32)
    concept_mean = cov_mean.type(torch.float32)
    return concept_vector, concept_mean

# def cal_concept(covs, means):
#     """
#     Extract concept vectors from covariance matrices
#     """
#     # Ensure all tensors are on the same device
#     device = covs.device

#     concept_vectors = []
#     concept_means = []

#     for j in range(covs.shape[0]):
#         # Original code might have DDP-specific operations
#         # We'll ensure everything stays on the same device

#         # Extract eigenvectors (concept vectors)
#         eigenvalues, eigenvectors = torch.linalg.eigh(covs[j])

#         # Sort eigenvectors by eigenvalues (descending)
#         sorted_indices = torch.argsort(eigenvalues, descending=True)
#         eigenvectors = eigenvectors[:, sorted_indices]

#         # Take the first eigenvector as the concept vector
#         concept_vector = eigenvectors[:, 0]
#         concept_vectors.append(concept_vector)

#         # Store corresponding mean
#         concept_means.append(means[j])

#     return torch.stack(concept_vectors), torch.stack(concept_means)


# def cal_class_MCP(model, concept_vecs, concept_means, dataloader, num_class, args):
#     if args.global_rank in [-1, 0]:
#         print("Calculate class MCP distributions!")
        
#     # class_count = torch.zeros(num_class).cuda(args.global_rank)

#     # device = args.device_id if args.device_id is not None and args.device_id != -1 else 'cpu'
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     class_count = torch.zeros(num_class, device=device)


#     # # print(class_count)
#     # class_node_resps = [torch.zeros([num_class] + [concept_vecs[0].shape[0]], requires_grad = False).cuda(args.global_rank),
#     #                     torch.zeros([num_class] + [concept_vecs[1].shape[0]], requires_grad = False).cuda(args.global_rank),
#     #                     torch.zeros([num_class] + [concept_vecs[2].shape[0]], requires_grad = False).cuda(args.global_rank),
#     #                     torch.zeros([num_class] + [concept_vecs[3].shape[0]], requires_grad = False).cuda(args.global_rank)]

#     class_node_resps = [torch.zeros([num_class] + [concept_vecs[0].shape[0]], requires_grad = False).cuda(args.global_rank).to(device),
#                         torch.zeros([num_class] + [concept_vecs[1].shape[0]], requires_grad = False).cuda(args.global_rank).to(device),
#                         torch.zeros([num_class] + [concept_vecs[2].shape[0]], requires_grad = False).cuda(args.global_rank).to(device),
#                         torch.zeros([num_class] + [concept_vecs[3].shape[0]], requires_grad = False).cuda(args.global_rank).to(device)]
    
#     pbar = enumerate(dataloader)
#     if args.global_rank in [-1, 0]:
#         pbar = tqdm.tqdm(pbar, total = len(dataloader))
        
#     with torch.no_grad():
#         for iteration, (img, label) in pbar:
#             class_count.index_add_(0, label.cuda(args.global_rank), torch.tensor([1.] * label.shape[0]).cuda(args.global_rank))
#             max_responses = []
#             img = img.cuda(args.global_rank)
#             l1, l2, l3, l4 = model(img)
#             feats = [l1, l2, l3, l4]
#             for layer_i, feat in enumerate(feats):
#                 B, C, H, W = feat.shape
#                 feat = feat.reshape(B, C // args.concept_cha[layer_i], args.concept_cha[layer_i], H, W)
#                 feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3).unsqueeze(4)
#                 feat_norm = feat / (torch.norm(feat, dim = 2, keepdim = True) + 1e-16)
#                 # calculate concept vector from covariance matrix
#                 concept_vector = concept_vecs[layer_i].cuda(args.global_rank)
#                 response = torch.sum(feat_norm * concept_vector.unsqueeze(0).unsqueeze(3).unsqueeze(4), dim = 2)
#                 max_response, max_index = torch.nn.functional.adaptive_max_pool2d(response, output_size = 1, return_indices = True)
#                 max_response = max_response[..., 0, 0]
                
#                 max_responses.append((max_response + 1) / 2)
#                 assert max_responses[-1].min() >= 0, "Response gets negative !!"
#                 max_index = max_index.repeat(1, 1, args.concept_cha[layer_i], 1)
#                 class_node_resps[layer_i].index_add_(0, label.cuda(args.global_rank), torch.clip(max_responses[layer_i], min = 1e-8))

#     dist.all_reduce(class_count)
#     for layer_i in range(len(class_node_resps)):
#         dist.all_reduce(class_node_resps[layer_i])
#         class_node_resps[layer_i] = class_node_resps[layer_i] / class_count[:, None]
    
#     class_node_resps = torch.cat(class_node_resps, dim = -1)
#     if args.global_rank in [-1, 0]:
#         print("class_node_resps : ", class_node_resps.shape)
#     class_node_resps = class_node_resps / torch.sum(class_node_resps, dim = 1, keepdim = True)
#     torch.cuda.empty_cache()
#     return class_node_resps

def cal_class_MCP(model, concept_vecs, concept_means, dataloader, num_class, args):
    if args.global_rank in [-1, 0]:
        print("Calculate class MCP distributions!")

    # Xác định device một cách nhất quán
    device = args.device_id if args.device_id is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Khởi tạo class_count trên device đúng
    class_count = torch.zeros(num_class, device=device)

    # Khởi tạo class_node_resps trên device đúng
    class_node_resps = [
        torch.zeros([num_class, concept_vecs[0].shape[0]], requires_grad=False, device=device),
        torch.zeros([num_class, concept_vecs[1].shape[0]], requires_grad=False, device=device),
        torch.zeros([num_class, concept_vecs[2].shape[0]], requires_grad=False, device=device),
        torch.zeros([num_class, concept_vecs[3].shape[0]], requires_grad=False, device=device)
    ]

    pbar = enumerate(dataloader)
    if args.global_rank in [-1, 0]:
        pbar = tqdm.tqdm(pbar, total=len(dataloader))

    with torch.no_grad():
        for iteration, (img, label) in pbar:
            # Chuyển dữ liệu sang device
            img = img.to(device)
            label = label.to(device)

            # Cập nhật class_count
            class_count.index_add_(0, label, torch.ones(label.shape[0], device=device))

            # Forward pass
            l1, l2, l3, l4 = model(img)
            feats = [l1, l2, l3, l4]

            max_responses = []
            for layer_i, feat in enumerate(feats):
                B, C, H, W = feat.shape
                feat = feat.reshape(B, C // args.concept_cha[layer_i], args.concept_cha[layer_i], H, W)

                # Đảm bảo concept_means ở trên device đúng
                concept_mean = concept_means[layer_i].to(device)
                feat = feat - concept_mean.unsqueeze(0).unsqueeze(3).unsqueeze(4)

                feat_norm = feat / (torch.norm(feat, dim=2, keepdim=True) + 1e-16)

                # Đảm bảo concept_vector ở trên device đúng
                concept_vector = concept_vecs[layer_i].to(device)
                response = torch.sum(feat_norm * concept_vector.unsqueeze(0).unsqueeze(3).unsqueeze(4), dim=2)

                max_response, max_index = torch.nn.functional.adaptive_max_pool2d(response, output_size=1, return_indices=True)
                max_response = max_response[..., 0, 0]

                max_responses.append((max_response + 1) / 2)
                assert max_responses[-1].min() >= 0, "Response gets negative !!"

                # Cập nhật class_node_resps
                class_node_resps[layer_i].index_add_(0, label, torch.clip(max_responses[layer_i], min=1e-8))

    # Trong môi trường single GPU, không cần all_reduce
    # Chỉ cần chuẩn hóa kết quả
    for layer_i in range(len(class_node_resps)):
        # Tránh chia cho 0
        valid_counts = class_count.clone()
        valid_counts[valid_counts == 0] = 1  # Tránh chia cho 0
        class_node_resps[layer_i] = class_node_resps[layer_i] / valid_counts.unsqueeze(1)

    # Ghép các layer lại
    class_node_resps = torch.cat(class_node_resps, dim=-1)

    if args.global_rank in [-1, 0]:
        print("class_node_resps : ", class_node_resps.shape)

    # Chuẩn hóa tổng
    sum_resps = torch.sum(class_node_resps, dim=1, keepdim=True)
    # Tránh chia cho 0
    sum_resps[sum_resps == 0] = 1
    class_node_resps = class_node_resps / sum_resps

    torch.cuda.empty_cache()
    return class_node_resps


def KL_div(x, y):
    return torch.sum(x * (torch.log2(x) - torch.log2(y)), dim = -1)

def JS_div(x, y):
    return (KL_div(x, (x + y) / 2) + KL_div(y, (x + y) / 2)) / 2

# def cal_JS_sim(img_MCP_dist, class_MCP_dist):
#     assert len(img_MCP_dist.shape) == 2 and len(class_MCP_dist.shape) == 1, f"Error shape of img_MCP_dist {img_MCP_dist.shape} and class_MCP_dist {class_MCP_dist.shape}!"
#     return JS_div(img_MCP_dist + 1e-8, class_MCP_dist.unsqueeze(0) + 1e-8)

def cal_JS_sim(x, y):
    assert x.shape[-1] != 1, "Wrong dimention of x!"
    assert y.shape[-1] != 1, "Wrong dimention of y!"
    assert x.shape[-1] == y.shape[-1], f"Miss match the dimension get {x.shape[-1]} and {y.shape[-1]}."
    M = (x + y) / 2    
    return (KL_div(x, M) + KL_div(y, M)) / 2

def cal_sim(img_MCP_dist, class_MCP_dist):
    feat_sim = cal_JS_sim(img_MCP_dist, class_MCP_dist)
    return feat_sim

def cal_acc(feats, class_MCP, concept_vecs, concept_means, args):
    max_responses = []
    for layer_i, feat in enumerate(feats):
        B, C, H, W = feat.shape
        feat = feat.reshape(B, C // args.concept_cha[layer_i], args.concept_cha[layer_i], H, W)
        feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3).unsqueeze(4)
        feat_norm = feat / (torch.norm(feat, dim = 2, keepdim = True) + 1e-16)
    
        # calculate concept vector from covariance matrix
        concept_vector = concept_vecs[layer_i].cuda(args.global_rank)
        response = torch.sum(feat_norm * concept_vector.unsqueeze(0).unsqueeze(3).unsqueeze(4), dim = 2)
        max_response, max_index = torch.nn.functional.adaptive_max_pool2d(response, output_size = 1, return_indices = True)
        max_responses.append((max_response[..., 0, 0] + 1) / 2)

    Diff_centroid_dist_resp = []
    max_responses = torch.cat(max_responses, dim = 1)
    max_responses = max_responses / torch.sum(max_responses, dim = 1, keepdim = True)
    for class_i in range(args.category):
        resp_sim = cal_sim(max_responses, class_MCP[class_i].unsqueeze(0))
        Diff_centroid_dist_resp.append(resp_sim)

    Diff_centroid_dist_resp = torch.stack(Diff_centroid_dist_resp, dim = 1)
    return torch.topk(-Diff_centroid_dist_resp, dim = 1, k = 1)[1], \
           torch.topk(-Diff_centroid_dist_resp, dim = 1, k = 5)[1]

# def cal_acc(features, class_MCP, concept_vectors, concept_means, args):
#     """
#     Calculate accuracy based on MCP prediction
#     """
#     # Ensure all tensors are on the same device
#     device = args.device_id

#     batch_size = features[0].shape[0]
#     num_classes = class_MCP[0].shape[0]

#     # Initialize scores for each class
#     scores = torch.zeros((batch_size, num_classes), device=device)

#     # Calculate scores based on MCP similarity
#     for i, feature in enumerate(features):
#         for j in range(args.concept_per_layer[i]):
#             # Project feature onto concept vector
#             concept_vector = concept_vectors[i][j].to(device)
#             concept_mean = concept_means[i][j].to(device)

#             # Calculate projection
#             centered_feature = feature - concept_mean.view(1, -1)
#             projection = torch.matmul(centered_feature, concept_vector)

#             # Calculate similarity to class MCP values
#             for c in range(num_classes):
#                 class_proj = class_MCP[i][c, j]
#                 similarity = -torch.abs(projection - class_proj)
#                 scores[:, c] += similarity

#     # Get top-k predictions
#     _, top1_indices = torch.topk(scores, k=1, dim=1)
#     _, top5_indices = torch.topk(scores, k=5, dim=1)

#     return top1_indices, top5_indices

def get_dataset(case_name: str) -> Tuple[str, str, str, int]:
    if "AWA2" in case_name:
        print("Using AWA2")
        data_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/"
        train_path = "train"
        val_path = "val"
        num_class = 50
    elif "CUB" in case_name:
        print("Using CUB")
        data_path = "/eva_data_4/bor/datasets/CUB_200_2011/"
        train_path = "train"
        val_path = "val"
        num_class = 200
    elif "Stanford" in case_name:
        print("Using StanfordCar")
        data_path = "/eva_data_4/bor/datasets/stanford car/"
        train_path = "cars_train"
        val_path = "cars_test"
        num_class = 196
    elif "Caltech101" in case_name:
        print("Using Caltech101")
        data_path = "./dataset/caltech_101/"
        train_path = "train"
        val_path = "test"
        num_class = 101
    elif "Food" in case_name:
        print("Using Food101")
        data_path = "/eva_data_4/bor/datasets/food-101/"
        train_path = "train"
        val_path = "test"
        num_class = 101
    elif "SYN" in case_name:
        print("Using SYN")
        data_path = "/eva_data_4/bor/datasets/Synthetic/"
        train_path = "train/raw"
        val_path = "val/raw"
        num_class = 15
    elif "ImageNet" in case_name:
        print("Using SYN")
        data_path = "/eva_data_4/bor/datasets/ImageNet2012/"
        train_path = "train_sampled"
        val_path = "val"
        num_class = 1000
    
    print("data path:", data_path)

    return data_path, train_path, val_path, num_class

def get_model_set(args):
    if args.basic_model == "resnet50":
        image_size = 224
    elif args.basic_model == "resnet34":
        image_size = 224
    elif args.basic_model == "resnet50_relu":
        image_size = 224
    elif args.basic_model == "vgg13":
        image_size = 224
    elif args.basic_model == "inceptionv3":
        image_size = 299
    elif args.basic_model == "mobilenet":
        image_size = 224
    elif args.basic_model == "mobilenet_relu":
        image_size = 224
    elif args.basic_model == "densenet":
        image_size = 224
    elif args.basic_model == "convnext_tiny":
        image_size = 224
    if args.basic_model == "vit_b_16":
        image_size = 224
    return args, image_size

def load_weight(model: torch.nn.Module, path: str, use_teacher: bool = False) -> None:
    print(f"load weight from {path}")
    print("CUDA is availble: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_teacher:
        parameters = torch.load(path, map_location={'cuda:1':'cuda:0'})["Teacher"]
    else:
        parameters = torch.load(path, map_location={'cuda:1':'cuda:0'})["Model"]
        # parameters = torch.load(path, map_location=torch.device('cpu'))["Model"]
    # model = torch.nn.DataParallel(model, device_ids = [0])
    model.to(device)
    model.load_state_dict(parameters, strict = True)

def load_model(model, basic_model, num_class) -> torch.nn.Module:
    if "resnet" in model.lower():
        model_class = importlib.import_module("ResNet")
    elif "inception" in model.lower():
        model_class = importlib.import_module("inception_net")
    elif "mobilenet" in model.lower():
        model_class = importlib.import_module("mobilenet")
    elif "convnext" in model.lower():
        model_class = importlib.import_module("convnext")
    else:
        model_class = importlib.import_module(model)
    model = model_class.load_model(basic_model, num_class)
    return model
    
def id2name(idx, conecept_per_layer):
    idx += 1
    layer_i = 1
    while idx > conecept_per_layer[layer_i - 1]:
        idx -= conecept_per_layer[layer_i - 1]
        layer_i += 1

    return f"l{layer_i}_{idx}"

def name2id(idx, concept_per_layer):
    layer = int(idx.split("_")[0][1:])
    i_th = int(idx.split("_")[1])
    count = 0
    for i in range(layer - 1):
        count += concept_per_layer[i]

    return count + i_th - 1