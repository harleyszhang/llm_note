import torch

target_device = "cpu"

def merge_input_ids_with_image_features(
    input_ids: torch.Tensor, 
    inputs_embeds: torch.Tensor, 
    image_features: torch.Tensor,
    pad_token_id: int,
    image_token_index: int
):
    """
    将 input_ids 与 image_features 合并，生成最终的嵌入和位置 ID。
    
    Args:
        input_ids (torch.Tensor): 输入的 token IDs, 形状为 (batch_size, sequence_length)
        inputs_embeds (torch.Tensor): 文本嵌入，形状为 (batch_size, sequence_length, embed_dim)
        image_features (torch.Tensor): 视觉编码后的图像特征，形状为 (num_images, num_image_patches, embed_dim)
        pad_token_id (int): 填充 token 的 ID
        image_token_index (int): 图像 token 的 ID
    
    Returns:
        final_embedding (torch.Tensor): 合并后的嵌入张量，形状为 (batch_size, max_embed_dim, embed_dim)
        position_ids (torch.Tensor): 位置 ID, 形状为 (batch_size, max_embed_dim)
    """
    # 1, 基础 shape 信息提取
    num_images, num_image_patches, embed_dim = image_features.shape # torch.Size([1, 576, 4096])
    batch_size, sequence_length = input_ids.shape # torch.Size([1, 22])

    # 2, 掩码与填充处理
    attention_mask = (input_ids != pad_token_id).long()
    left_padding = not torch.sum(input_ids[:, -1] == pad_token_id).bool().any()
    
    special_image_token_mask = input_ids == image_token_index
    
    # 3, 计算新序列长度
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
    batch_indices, non_image_indices = torch.where(input_ids != image_token_index) 

    # 4, 位置映射计算
    # 得到每个原始 token 在新序列中占据的开始位置索引。
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1 
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1] 
    if left_padding:
        new_token_positions += nb_image_pad[:, None]  # offset for left padding
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
    
    # 5，构建融合张量
    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices] # 填充文本嵌入

    # 确定图像特征插入位置，通过找到 final_embedding 中所有全 0 的位置
    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)  # 找出 final_embedding 中所有维度为0的位置
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

    # 将 image_features 重新排列为 (num_images * num_image_patches, embed_dim)，并填充到 final_embedding 的相应位置。
    final_embedding[image_to_overwrite] = image_features.contiguous().view(-1, embed_dim).to(target_device)
    
    # 6，生成新的 position_ids
    position_ids = torch.arange(max_embed_dim, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

    # 7，处理填充位置的嵌入, 将填充位置的嵌入设为0
    batch_indices_pad, pad_indices = torch.where(input_ids == pad_token_id)
    indices_to_mask = new_token_positions[batch_indices_pad, pad_indices]

    final_embedding[batch_indices_pad, indices_to_mask] = 0

    return final_embedding, position_ids

# === 输入示例 ===
# batch_size=1, seq_len=5, embed_dim=3; num_images=1, num_patches=2
input_ids       = torch.tensor([[11, 99, 22, 0, 0]])       # 99 代表 image_token_index，0 是 pad_token_id
inputs_embeds   = torch.arange(1, 1+1*5*3).reshape(1,5,3).float()
image_features  = torch.tensor([[[9,9,9],[8,8,8]]]).float()  # shape (1,2,3)
pad_token_id    = 0
image_token_index = 99

fe, pids = merge_input_ids_with_image_features(
    input_ids,
    inputs_embeds,
    image_features,
    pad_token_id,
    image_token_index
)

print("final_embedding shape:", fe.shape)
print("position_ids      shape:", pids.shape)
print("final_embedding:\n", fe)
print("position_ids:\n", pids)