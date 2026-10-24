use burn::prelude::*;
use dre_core::BatchLayout;

/// Streaming Burn tensor-op oracle used for comparison with the CubeCL implementation.
pub(crate) fn oracle_windowed_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, layout: &BatchLayout, radius: usize) -> Tensor<4> {
    let [batch, heads, tokens, width] = q.dims();
    let mut document_ids = vec![vec![None; tokens]; batch];
    for (b, docs) in layout.host.documents_per_row().iter().enumerate() {
        let mut start = 0;
        for (id, &length) in docs.iter().enumerate() {
            for position in start..start + length { document_ids[b][position] = Some(id); }
            start += length;
        }
    }
    let device = q.device();
    let mut maximum = Tensor::zeros([batch, heads, tokens, 1], &device);
    let mut normalizer = Tensor::zeros([batch, heads, tokens, 1], &device);
    let mut accumulator = Tensor::zeros([batch, heads, tokens, width], &device);
    for offset in -(radius as isize)..=radius as isize {
        let query_start = if offset < 0 { (-offset) as usize } else { 0 };
        let query_end = if offset > 0 { tokens.saturating_sub(offset as usize) } else { tokens };
        if query_start >= query_end { continue; }
        let key_start = (query_start as isize + offset) as usize;
        let key_end = key_start + query_end - query_start;
        let q_piece = q.clone().slice([0..batch, 0..heads, query_start..query_end, 0..width]);
        let k_piece = k.clone().slice([0..batch, 0..heads, key_start..key_end, 0..width]);
        let v_piece = v.clone().slice([0..batch, 0..heads, key_start..key_end, 0..width]);
        let mut valid = Vec::with_capacity(batch * (query_end - query_start));
        for b in 0..batch {
            for query in query_start..query_end {
                let key = (query as isize + offset) as usize;
                let same_document = document_ids[b][query].is_some() && document_ids[b][query] == document_ids[b][key];
                valid.push(same_document);
            }
        }
        let valid = Tensor::<3, Bool>::from_data(TensorData::new(valid, [batch, query_end - query_start, 1]), &device)
            .reshape([batch, 1, query_end - query_start, 1])
            .expand([batch, heads, query_end - query_start, 1]);
        let score = (q_piece * k_piece).sum_dim(3) * (1.0f32 / (width as f32).sqrt());
        let old_max = maximum.clone().slice([0..batch, 0..heads, query_start..query_end, 0..1]);
        let old_sum = normalizer.clone().slice([0..batch, 0..heads, query_start..query_end, 0..1]);
        let first = old_sum.clone().equal_elem(0.0);
        let new_max = old_max.clone().max_pair(score.clone())
            .mask_where(first.clone(), score.clone())
            .mask_where(valid.clone().bool_not(), old_max.clone());
        // Avoid evaluating exp(0 - first_score) before multiplying by the empty accumulator.
        let alpha = (old_max - new_max.clone()).mask_fill(first, 0.0).exp();
        let probability = (score - new_max.clone())
            .mask_fill(valid.clone().bool_not(), 0.0)
            .exp()
            .mask_fill(valid.bool_not(), 0.0);
        let new_sum = alpha.clone() * old_sum + probability.clone();
        let old_acc = accumulator.clone().slice([0..batch, 0..heads, query_start..query_end, 0..width]);
        let new_acc = old_acc * alpha.expand([batch, heads, query_end - query_start, width])
            + v_piece * probability.expand([batch, heads, query_end - query_start, width]);
        maximum = maximum.slice_assign([0..batch, 0..heads, query_start..query_end, 0..1], new_max);
        normalizer = normalizer.slice_assign([0..batch, 0..heads, query_start..query_end, 0..1], new_sum);
        accumulator = accumulator.slice_assign([0..batch, 0..heads, query_start..query_end, 0..width], new_acc);
    }
    // Padding has no eligible keys, so keep its zero accumulator and avoid 0 / 0.
    let empty = normalizer.clone().equal_elem(0.0);
    accumulator / normalizer.mask_fill(empty, 1.0).expand([batch, heads, tokens, width])
}

#[cfg(any(test, feature = "cpu-oracle"))]
pub(crate) fn burn_source_oracle(
    q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, relay_k: Tensor<4>, relay_v: Tensor<4>, layout: &BatchLayout,
) -> Tensor<4> {
    let [batch, heads, tokens, width] = q.dims();
    let mut output = Tensor::zeros([batch, heads, tokens, width], &q.device());
    let mut document_index = 0usize;
    for (row, documents) in layout.host.documents_per_row().iter().enumerate() {
        let mut start = 0usize;
        for &length in documents {
            let end = start + length;
            let source_k = k.clone().slice([row..row + 1, 0..heads, start..end, 0..width]);
            let source_v = v.clone().slice([row..row + 1, 0..heads, start..end, 0..width]);
            let global_k = relay_k.clone().slice([document_index..document_index + 1, 0..heads, 0..16, 0..width]);
            let global_v = relay_v.clone().slice([document_index..document_index + 1, 0..heads, 0..16, 0..width]);
            let mut rows = Vec::with_capacity(length);
            for token in 0..length {
                let query = q.clone().slice([row..row + 1, 0..heads, start + token..start + token + 1, 0..width]);
                let lo = token.saturating_sub(128);
                let hi = (token + 129).min(length);
                let local_k = source_k.clone().slice([0..1, 0..heads, lo..hi, 0..width]);
                let local_v = source_v.clone().slice([0..1, 0..heads, lo..hi, 0..width]);
                rows.push(burn::tensor::module::attention(query, Tensor::cat(vec![local_k, global_k.clone()], 2), Tensor::cat(vec![local_v, global_v.clone()], 2), None, None, burn::tensor::ops::AttentionModuleOptions::default()));
            }
            output = output.slice_assign([row..row + 1, 0..heads, start..end, 0..width], Tensor::cat(rows, 2));
            document_index += 1;
            start = end;
        }
    }
    output.mask_fill(layout.valid.clone().equal_elem(0.0).reshape([batch, 1, tokens, 1]), 0.0)
}
