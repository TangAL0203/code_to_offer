import torch
import torch.nn as nn

class transformer(nn.Module):
    def __init__(self):

        # 输入x token的embeddings，三个不同的fc，矩阵得到q,k,v
        x = embeddings + pos_encoding
        q, k, v = fc1(x), fc2(x), fc3(x)
        attention_weights = q * k^ / sqrt(ndim)
        ## attention_weights * = mask
        drop_out(attention_weights)
        out = attention_weights * v
        out = ffn(out)
        return out


        # 第一阶段： 计算得到Q、K、V
        q = F.linear(query, q_proj_weight)
        #  [tgt_len,batch_size,embed_dim] x [embed_dim,kdim * num_heads]
        #  = [tgt_len,batch_size,kdim * num_heads]
        k = F.linear(key, k_proj_weight)
        # [src_len, batch_size,embed_dim] x [embed_dim,kdim * num_heads]
        # = [src_len,batch_size,kdim * num_heads]
        v = F.linear(value, v_proj_weight)
        # [src_len, batch_size,embed_dim] x [embed_dim,vdim * num_heads]
        # = [src_len,batch_size,vdim * num_heads]

        # 第三阶段： 计算得到注意力权重矩阵
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        # [batch_size * num_heads,tgt_len,kdim]
        # 因为前面是num_heads个头一起参与的计算，所以这里要进行一下变形，以便于后面计算。 且同时交换了0，1两个维度
        k = k.contiguous().view(-1, bsz*num_heads, head_dim).transpose(0,1)
        #[batch_size * num_heads,src_len,kdim]
        v = v.contiguous().view(-1, bsz*num_heads, head_dim).transpose(0,1)
        #[batch_size * num_heads,src_len,vdim]
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))


        # 第四阶段： 进行相关掩码操作（decoder有）
        if attn_mask is not None:
            attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            # 变成 [batch_size, num_heads, tgt_len, src_len]的形状
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
             # 扩展维度，从[batch_size,src_len]变成[batch_size,1,1,src_len]
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,src_len)
            # [batch_size * num_heads, tgt_len, src_len]