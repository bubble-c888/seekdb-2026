/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX STORAGE

#include "ob_text_daat_iter.h"

namespace oceanbase
{
namespace storage
{
int ObTextDaaTIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRDaaTIterImpl::init(*param.base_param_, *param.dim_iters_,
                                            *param.allocator_, *param.relevance_collector_))) {
  } else if (param.need_relevance_ && OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  } else {
    mode_flag_ = param.mode_flag_;
    function_lookup_mode_ = param.function_lookup_mode_;
    need_relevance_ = param.need_relevance_;
    direct_union_mode_checked_ = false;
    direct_union_mode_ = false;
    direct_union_cursors_ = nullptr;
    direct_union_out_ids_ = nullptr;
    direct_union_out_bytes_ = nullptr;
    // same ordering as ObSRMergeCmp, with fast paths for the common doc id types
    const sql::ObDatumMeta &id_meta = param.base_param_->id_proj_expr_->datum_meta_;
    if (ObIntType == id_meta.type_) {
      union_cmp_type_ = UNION_ID_CMP_INT;
    } else if (ObUInt64Type == id_meta.type_) {
      union_cmp_type_ = UNION_ID_CMP_UINT;
    } else if (ObVarcharType == id_meta.type_ && CS_TYPE_BINARY == id_meta.cs_type_) {
      union_cmp_type_ = UNION_ID_CMP_BINARY;
    } else {
      union_cmp_type_ = UNION_ID_CMP_FUNC;
    }
    common::ObDatumBasicFuncs *basic_funcs = ObDatumFuncs::get_basic_func(id_meta.type_, id_meta.cs_type_);
    if (OB_ISNULL(basic_funcs) || OB_ISNULL(union_cmp_func_ = basic_funcs->null_first_cmp_)) {
      ret = OB_ERR_UNEXPECTED;
    }
  }
  return ret;
}

int ObTextDaaTIter::compare_union_id(const ObDatum &l, const ObDatum &r, int &cmp_ret) const
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(l.is_null() || r.is_null())) {
    ret = union_cmp_func_(l, r, cmp_ret, nullptr);
  } else {
    switch (union_cmp_type_) {
      case UNION_ID_CMP_INT: {
        const int64_t lv = l.get_int();
        const int64_t rv = r.get_int();
        cmp_ret = lv < rv ? -1 : (lv > rv ? 1 : 0);
        break;
      }
      case UNION_ID_CMP_UINT: {
        const uint64_t lv = l.get_uint64();
        const uint64_t rv = r.get_uint64();
        cmp_ret = lv < rv ? -1 : (lv > rv ? 1 : 0);
        break;
      }
      case UNION_ID_CMP_BINARY: {
        const int64_t min_len = MIN(l.len_, r.len_);
        const int res = MEMCMP(l.ptr_, r.ptr_, min_len);
        cmp_ret = 0 != res ? (res < 0 ? -1 : 1)
                           : (l.len_ < r.len_ ? -1 : (l.len_ > r.len_ ? 1 : 0));
        break;
      }
      default: {
        ret = union_cmp_func_(l, r, cmp_ret, nullptr);
        break;
      }
    }
  }
  return ret;
}

int ObTextDaaTIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (need_relevance_ || function_lookup_mode_
      || sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE != mode_flag_
      || iter_param_->need_project_relevance() || iter_param_->need_filter()) {
    ret = ObSRDaaTIterImpl::get_next_rows(capacity, count);
  } else if (!direct_union_mode_checked_ && OB_FAIL(check_direct_union_mode())) {
  } else if (direct_union_mode_) {
    ret = direct_union_next_rows(capacity, count);
  } else {
    ret = union_next_rows(capacity, count);
  }
  return ret;
}

int ObTextDaaTIter::check_direct_union_mode()
{
  int ret = OB_SUCCESS;
  const int64_t dim_cnt = dim_iters_->count();
  const int64_t max_batch_size = MAX(iter_param_->eval_ctx_->max_batch_size_, 1L);
  direct_union_mode_ = dim_cnt > 0
      && !need_relevance_
      && !function_lookup_mode_
      && sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE == mode_flag_
      && !iter_param_->need_project_relevance()
      && !iter_param_->need_filter()
      && iter_param_->eval_ctx_->is_vectorized()
      && ObTextDocIdKey::is_supported(iter_param_->id_proj_expr_->datum_meta_);
  if (!direct_union_mode_) {
  } else if (OB_ISNULL(direct_union_cursors_ = static_cast<DirectUnionCursor *>(
                 iter_allocator_->alloc(sizeof(DirectUnionCursor) * dim_cnt)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_ISNULL(direct_union_out_ids_ = static_cast<ObTextDocIdKey *>(
                 iter_allocator_->alloc(sizeof(ObTextDocIdKey) * max_batch_size)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (common::ObUInt64Type != iter_param_->id_proj_expr_->datum_meta_.type_
             && OB_ISNULL(direct_union_out_bytes_ = static_cast<char *>(
                 iter_allocator_->alloc(ObTextDocIdKey::BINARY_ID_LEN * max_batch_size)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    reset_direct_union_cursors();
  }
  if (OB_SUCC(ret)) {
    direct_union_mode_checked_ = true;
  } else {
    direct_union_mode_ = false;
  }
  return ret;
}

void ObTextDaaTIter::reset_direct_union_cursors()
{
  if (OB_NOT_NULL(direct_union_cursors_) && OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      direct_union_cursors_[i].keys_ = nullptr;
      direct_union_cursors_[i].count_ = 0;
      direct_union_cursors_[i].pos_ = 0;
      direct_union_cursors_[i].end_ = false;
    }
  }
}

int ObTextDaaTIter::load_direct_union_cursor(DirectUnionCursor &cursor, const int64_t dim_idx)
{
  int ret = OB_SUCCESS;
  cursor.pos_ = 0;
  cursor.count_ = 0;
  while (OB_SUCC(ret) && 0 == cursor.count_) {
    if (OB_FAIL(static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(dim_idx))
                ->get_next_id_key_batch(cursor.keys_, cursor.count_))) {
      if (OB_ITER_END == ret) {
        cursor.end_ = true;
        cursor.count_ = 0;
        ret = OB_SUCCESS;
        break;
      }
    }
  }
  return ret;
}

int ObTextDaaTIter::direct_union_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  count = 0;
  const int64_t dim_cnt = dim_iters_->count();
  const int64_t real_capacity = MIN(capacity, iter_param_->max_batch_size_);
  const common::ObLimitParam *limit_param = iter_param_->limit_param_;
  const bool has_limit = limit_param->is_valid();
  const int64_t should_match = static_cast<ObSRDaaTInnerProductRelevanceCollector *>(
      relevance_collector_)->minimum_should_match();
  for (int64_t i = 0; OB_SUCC(ret) && i < dim_cnt; ++i) {
    DirectUnionCursor &cursor = direct_union_cursors_[i];
    if (!cursor.end_ && cursor.pos_ >= cursor.count_ && OB_FAIL(load_direct_union_cursor(cursor, i))) {
    }
    if (OB_ITER_END == ret) {
      ret = OB_SUCCESS;
    }
  }
  while (OB_SUCC(ret) && count < real_capacity) {
    ObTextDocIdKey min_id = {0, 0};
    bool found = false;
    for (int64_t i = 0; i < dim_cnt; ++i) {
      const DirectUnionCursor &cursor = direct_union_cursors_[i];
      if (!cursor.end_) {
        const ObTextDocIdKey &id = cursor.keys_[cursor.pos_];
        if (!found || id < min_id) {
          min_id = id;
          found = true;
        }
      }
    }
    if (!found) {
      ret = OB_ITER_END;
    } else {
      int64_t matched_cnt = 0;
      for (int64_t i = 0; OB_SUCC(ret) && i < dim_cnt; ++i) {
        DirectUnionCursor &cursor = direct_union_cursors_[i];
        if (!cursor.end_ && cursor.keys_[cursor.pos_] == min_id) {
          ++matched_cnt;
          if (++cursor.pos_ >= cursor.count_ && OB_FAIL(load_direct_union_cursor(cursor, i))) {
            if (OB_ITER_END == ret) {
              ret = OB_SUCCESS;
            }
          }
        }
      }
      if (OB_FAIL(ret) || matched_cnt < should_match) {
      } else if (FALSE_IT(++input_row_cnt_)) {
      } else if (has_limit && input_row_cnt_ <= limit_param->offset_) {
      } else {
        direct_union_out_ids_[count++] = min_id;
        ++output_row_cnt_;
        if (has_limit && output_row_cnt_ >= limit_param->limit_) {
          ret = OB_ITER_END;
        }
      }
    }
  }
  if ((OB_SUCC(ret) || OB_ITER_END == ret) && count > 0) {
    ObExpr *id_proj_expr = iter_param_->id_proj_expr_;
    ObEvalCtx *eval_ctx = iter_param_->eval_ctx_;
    ObEvalCtx::BatchInfoScopeGuard guard(*eval_ctx);
    sql::ObBitVector &id_evaluated_flags = id_proj_expr->get_evaluated_flags(*eval_ctx);
    for (int64_t i = 0; i < count; ++i) {
      guard.set_batch_idx(i);
      ObDatum &id_datum = id_proj_expr->locate_datum_for_write(*eval_ctx);
      if (OB_ISNULL(direct_union_out_bytes_)) {
        id_datum.set_uint(direct_union_out_ids_[i].lo_);
      } else {
        uint64_t hi = direct_union_out_ids_[i].hi_;
        uint64_t lo = direct_union_out_ids_[i].lo_;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        hi = __builtin_bswap64(hi);
        lo = __builtin_bswap64(lo);
#endif
        char *buf = direct_union_out_bytes_ + i * ObTextDocIdKey::BINARY_ID_LEN;
        MEMCPY(buf, &hi, sizeof(uint64_t));
        MEMCPY(buf + sizeof(uint64_t), &lo, sizeof(uint64_t));
        id_datum.set_string(buf, ObTextDocIdKey::BINARY_ID_LEN);
      }
      id_evaluated_flags.set(i);
    }
    id_proj_expr->set_evaluated_projected(*eval_ctx);
  }
  return ret;
}

int ObTextDaaTIter::union_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  count = 0;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(0 == capacity)) {
  } else if (0 == dim_iters_->count()) {
    ret = OB_ITER_END;
  } else if (iter_param_->limit_param_->is_valid() && output_row_cnt_ >= iter_param_->limit_param_->limit_) {
    ret = OB_ITER_END;
  } else {
    const int64_t dim_cnt = dim_iters_->count();
    const int64_t should_match = static_cast<ObSRDaaTInnerProductRelevanceCollector *>(
        relevance_collector_)->minimum_should_match();
    const int64_t real_capacity = MIN(capacity, iter_param_->max_batch_size_);
    while (OB_SUCC(ret) && count < real_capacity) {
      // advance the dimensions consumed by the previous output document
      for (int64_t i = 0; OB_SUCC(ret) && i < next_round_cnt_; ++i) {
        const int64_t iter_idx = next_round_iter_idxes_[i];
        ObTextRetrievalDaaTTokenIter *dim_iter =
            static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(iter_idx));
        if (dim_iter->advance_cached_id(iter_domain_ids_[iter_idx])) {
        } else if (OB_FAIL(dim_iter->get_next_row())) {
          if (OB_ITER_END == ret) {
            ret = OB_SUCCESS;
            iter_domain_ids_[iter_idx] = nullptr;
          }
        } else if (OB_FAIL(dim_iter->get_curr_id(iter_domain_ids_[iter_idx]))) {
        }
      }
      next_round_cnt_ = 0;
      // pick the smallest current doc id among the alive dimensions
      const ObDatum *min_id = nullptr;
      for (int64_t i = 0; OB_SUCC(ret) && i < dim_cnt; ++i) {
        const ObDatum *cur_id = iter_domain_ids_[i];
        int cmp_ret = 0;
        if (nullptr == cur_id) {
          continue;
        } else if (nullptr == min_id) {
          cmp_ret = -1;
        } else if (OB_FAIL(compare_union_id(*cur_id, *min_id, cmp_ret))) {
        }
        if (OB_FAIL(ret)) {
        } else if (cmp_ret < 0) {
          min_id = cur_id;
          next_round_cnt_ = 0;
          next_round_iter_idxes_[next_round_cnt_++] = i;
        } else if (0 == cmp_ret) {
          next_round_iter_idxes_[next_round_cnt_++] = i;
        }
      }
      if (OB_FAIL(ret)) {
      } else if (nullptr == min_id) {
        ret = OB_ITER_END;
      } else {
        // Natural-language score-free retrieval only needs the number of
        // matched dimensions, not a virtual relevance-collection round.
        const bool is_valid = next_round_cnt_ >= should_match;
        if (is_valid && OB_FAIL(cache_result(count, *min_id, 0.0))) {
          if (OB_UNLIKELY(OB_ITER_END != ret)) {
          }
        }
      }
    }
    if ((OB_SUCC(ret) || OB_ITER_END == ret) && count > 0) {
      const int tmp_ret = ret;
      if (OB_FAIL(project_results(count))) {
      } else {
        ret = tmp_ret;
      }
    }
  }
  return ret;
}

void ObTextDaaTIter::reset()
{
  direct_union_mode_checked_ = false;
  direct_union_mode_ = false;
  direct_union_cursors_ = nullptr;
  direct_union_out_ids_ = nullptr;
  direct_union_out_bytes_ = nullptr;
  bm25_param_estimator_.reset();
  // this class can know the type of dim_iters_, but ObSRDaaTIterImpl maybe not
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRDaaTIterImpl::reset();
}

void ObTextDaaTIter::reuse(const bool switch_tablet)
{
  if ((OB_NOT_NULL(dim_iters_) && 0 == dim_iters_->count())) {
    // do nothing
  } else {
    bm25_param_estimator_.reuse(switch_tablet);
  }
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reuse();
    }
  }
  reset_direct_union_cursors();
  ObSRDaaTIterImpl::reuse(switch_tablet);
}

int ObTextDaaTIter::pre_process()
{
  int ret = OB_SUCCESS;
  if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (need_relevance_ && iter_param_->need_project_relevance()) {
    if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
    }
  }
  return ret;
}

ObTextBMWIter::ObTextBMWIter()
  : ObSRBMWIterImpl(),
    bm25_param_estimator_() {}

void ObTextBMWIter::reuse(const bool switch_tablet)
{
  bm25_param_estimator_.reuse(switch_tablet);
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRBMWIterImpl::reuse(switch_tablet);
}

void ObTextBMWIter::reset()
{
  bm25_param_estimator_.reset();
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRBMWIterImpl::reset();
}

int ObTextBMWIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRBMWIterImpl::init(*param.base_param_, *param.dim_iters_,
                                           *param.allocator_, *param.relevance_collector_))) {
  } else if (OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  }
  return ret;
}

int ObTextBMWIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
  }

  if (FAILEDx(ObSRBMWIterImpl::get_next_rows(capacity, count))) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    }
  }
  return ret;
}

int ObTextBMWIter::init_before_wand_process()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!bm25_param_estimator_.is_estimated())) {
    ret = OB_ERR_UNEXPECTED;
  }

  for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
    ObTextRetrievalBlockMaxIter *block_max_iter = static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i));
    if (OB_FAIL(block_max_iter->init_block_max_iter(
        bm25_param_estimator_.get_total_doc_cnt(), bm25_param_estimator_.get_avg_doc_token_cnt()))) {
    }
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase
