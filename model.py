import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ==========================================
# 1. FAAM: 频域自适应增强模块
# Frequency-Domain Adaptive Augmentation Module
# ==========================================
class FAAM(nn.Module):
    def __init__(self, lam=0.3):
        super(FAAM, self).__init__()
        self.lam = lam # 对应论文中的 \lambda，控制SAR风格注入强度

    def forward(self, x_opt, x_sar):
        """
        x_opt: [B, C, H, W] 光学图像
        x_sar: [B, C, H, W] SAR 图像
        """
        # 1. 傅里叶变换 (Eq. 1)
        fft_opt = torch.fft.fft2(x_opt, norm="ortho")
        fft_sar = torch.fft.fft2(x_sar, norm="ortho")

        # 2. 分离振幅与相位 (Eq. 2)
        amp_opt, phase_opt = torch.abs(fft_opt), torch.angle(fft_opt)
        amp_sar = torch.abs(fft_sar)

        # 3. 跨模态振幅混合 (Eq. 3)
        amp_mix = (1 - self.lam) * amp_opt + self.lam * amp_sar

        # 4. 逆傅里叶变换重建 (Eq. 4)
        fft_mix = amp_mix * torch.exp(1j * phase_opt)
        x_opt_aug = torch.fft.ifft2(fft_mix, norm="ortho").real
        
        return x_opt_aug

# ==========================================
# 2. TGAM: 拓扑图对齐模块 & Sinkhorn OT Loss
# Topological Graph Alignment Module
# ==========================================
class TGAM(nn.Module):
    def __init__(self, embed_dim=768, num_nodes=6, knn=3, tau=0.1):
        super(TGAM, self).__init__()
        self.k = num_nodes
        self.knn = knn
        self.tau = tau
        
        # Adaptive part-based pooling: 将 N 个 token 聚类/映射为 k 个节点
        self.node_pool = nn.AdaptiveAvgPool1d(num_nodes)
        # GCN 权重 W^{(l)} (Eq. 8)
        self.gcn_weight = nn.Linear(embed_dim, embed_dim)

    def forward(self, features):
        """
        features: ViT 输出的 token 序列 [B, N, D]
        """
        B, N, D = features.shape
        
        # 1. 节点划分 (Node Partitioning): [B, N, D] -> [B, D, N] -> [B, D, k] -> [B, k, D]
        nodes = self.node_pool(features.transpose(1, 2)).transpose(1, 2)
        
        # 2. 边构建 (Edge Construction): 计算高斯亲和矩阵 S
        dist_matrix = torch.cdist(nodes, nodes) # [B, k, k]
        S = torch.exp(- (dist_matrix ** 2) / self.tau)
        
        # k-NN 掩码，保留最强连接
        topk_vals, topk_indices = torch.topk(S, self.knn, dim=-1)
        A = torch.zeros_like(S).scatter_(-1, topk_indices, topk_vals)
        
        # 3. GCN 聚合 (Eq. 8)
        I = torch.eye(self.k, device=features.device).unsqueeze(0)
        A_tilde = A + I
        D_tilde = torch.diag_embed(torch.sum(A_tilde, dim=-1) ** -0.5)
        
        # \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} V W
        norm_A = torch.bmm(torch.bmm(D_tilde, A_tilde), D_tilde)
        nodes_refined = F.relu(self.gcn_weight(torch.bmm(norm_A, nodes)))
        
        return nodes_refined

def sinkhorn_ot_loss(V_opt, V_sar, epsilon=0.05, n_iters=10):
    """
    计算基于 Sinkhorn 最优传输的图一致性损失 (Eq. 12)
    V_opt, V_sar: [B, k, D]
    """
    B, k, D = V_opt.shape
    
    # 计算余弦距离代价矩阵 C (Cost Matrix)
    V_opt_norm = F.normalize(V_opt, p=2, dim=-1)
    V_sar_norm = F.normalize(V_sar, p=2, dim=-1)
    C = 1.0 - torch.bmm(V_opt_norm, V_sar_norm.transpose(1, 2)) # [B, k, k]
    
    # Sinkhorn 迭代求解最优耦合矩阵 P
    K = torch.exp(-C / epsilon)
    u = torch.ones(B, k, device=V_opt.device) / k
    v = torch.ones(B, k, device=V_sar.device) / k
    
    for _ in range(n_iters):
        u = (1.0 / k) / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
        v = (1.0 / k) / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
    
    P_star = u.unsqueeze(-1) * K * v.unsqueeze(1)
    
    # 计算最终的图匹配损失: sum(P_star * C)
    loss_graph = torch.sum(P_star * C, dim=(1, 2)).mean()
    return loss_graph

# ==========================================
# 3. MDFE: 模态解耦特征提取器
# Modality-Disentangled Feature Extractor
# ==========================================
def orthogonality_loss(f_sh, f_sp):
    """
    序列级正交损失 (Eq. 7)
    f_sh, f_sp: [B, D] 或 [B, N, D]
    """
    if f_sh.dim() == 3:
        f_sh = f_sh.mean(dim=1) # 简化表示，实际可对逐个 token 算
        f_sp = f_sp.mean(dim=1)
    
    cos_sim = F.cosine_similarity(f_sh, f_sp, dim=-1)
    loss_orth = torch.mean(torch.abs(cos_sim))
    return loss_orth

class FDGNN_Model(nn.Module):
    def __init__(self, num_classes=640, embed_dim=768):
        super(FDGNN_Model, self).__init__()
        self.faam = FAAM(lam=0.3)
        self.tgam = TGAM(embed_dim=embed_dim, num_nodes=6)
        
        # 在 Hoss-ReID 中，您可以直接从 timm 库加载预训练的 ViT
        # 这里为了能够直接运行测试，使用简单的线性层替代完整的 ViT
        # 实际使用时：self.encoder_sh = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.encoder_sh = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(256 * 256, embed_dim), # 模拟 ViT token 映射
            nn.LayerNorm(embed_dim)          # 共享分支带有归一化
        )
        self.encoder_sp = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(256 * 256, embed_dim)  # 特定分支故意不带归一化
        )
        
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_opt, x_sar, is_train=True):
        if is_train:
            # 1. 频域增强
            x_opt_aug = self.faam(x_opt, x_sar)
            
            # 2. 共享与特定分支特征提取 (模拟输出 [B, N, D] 的 token 序列)
            # 注意：这里我们用维度扩张模拟 N=196 的 ViT sequence
            f_sh_opt = self.encoder_sh(x_opt_aug).unsqueeze(1).repeat(1, 196, 1) 
            f_sp_opt = self.encoder_sp(x_opt_aug).unsqueeze(1).repeat(1, 196, 1)
            
            f_sh_sar = self.encoder_sh(x_sar).unsqueeze(1).repeat(1, 196, 1)
            f_sp_sar = self.encoder_sp(x_sar).unsqueeze(1).repeat(1, 196, 1)
            
            # 3. TGAM 拓扑节点提取
            nodes_opt = self.tgam(f_sh_opt)
            nodes_sar = self.tgam(f_sh_sar)
            
            # 池化为全局特征用于分类
            global_feat_opt = f_sh_opt.mean(dim=1)
            global_feat_sar = f_sh_sar.mean(dim=1)
            
            logits_opt = self.classifier(global_feat_opt)
            logits_sar = self.classifier(global_feat_sar)
            
            return (logits_opt, logits_sar), (global_feat_opt, global_feat_sar), (f_sh_opt, f_sp_opt, f_sh_sar, f_sp_sar), (nodes_opt, nodes_sar)
        else:
            # 测试阶段只需提取共享特征
            f_sh = self.encoder_sh(x_opt).unsqueeze(1).repeat(1, 196, 1)
            return f_sh.mean(dim=1)

# ==========================================
# 4. 运行验证代码 (切实可运行部分)
# ==========================================
if __name__ == "__main__":
    print("Initializing FD-GNN Pipeline...")
    
    # 模拟输入参数: Batch_size=4, C=3, H=256, W=256
    B, C, H, W = 4, 3, 256, 256
    x_opt = torch.rand(B, C, H, W)
    x_sar = torch.rand(B, C, H, W)
    labels = torch.randint(0, 640, (B,))
    
    # 实例化模型
    model = FDGNN_Model()
    
    # 前向传播
    print("Running Forward Pass...")
    logits, global_feats, disentangle_feats, graph_nodes = model(x_opt, x_sar, is_train=True)
    
    # 损失计算
    print("Computing Losses...")
    # 1. 身份分类损失 (以光为例，使用交叉熵代替完整 Label Smoothing)
    loss_id = F.cross_entropy(logits[0], labels) + F.cross_entropy(logits[1], labels)
    
    # 2. 正交解耦损失 (Eq. 7)
    f_sh_opt, f_sp_opt, f_sh_sar, f_sp_sar = disentangle_feats
    loss_orth = orthogonality_loss(f_sh_opt, f_sp_opt) + orthogonality_loss(f_sh_sar, f_sp_sar)
    
    # 3. 拓扑图对齐损失 (Eq. 12)
    nodes_opt, nodes_sar = graph_nodes
    loss_graph = sinkhorn_ot_loss(nodes_opt, nodes_sar)
    
    # 总损失 (假定 alpha=0.1, gamma=0.5)
    loss_total = loss_id + 0.1 * loss_orth + 0.5 * loss_graph
    
    print("-" * 40)
    print(f"Classification Loss: {loss_id.item():.4f}")
    print(f"Orthogonal Loss:     {loss_orth.item():.4f}")
    print(f"Graph OT Loss:       {loss_graph.item():.4f}")
    print(f"Total Loss:          {loss_total.item():.4f}")
    print("-" * 40)
    
    # 验证反向传播是否联通
    loss_total.backward()
    print("Backward Pass Successful! Gradients are flowing.")
