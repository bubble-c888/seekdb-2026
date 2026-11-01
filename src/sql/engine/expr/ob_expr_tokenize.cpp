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

#include "sql/engine/expr/ob_expr_tokenize.h"

#include "lib/alloc/alloc_struct.h"
#include "lib/allocator/page_arena.h"
#include "lib/charset/ob_charset.h"
#include "common/json_type/ob_json_base.h"
#include "common/json_type/ob_json_tree.h"
#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/string/ob_string.h"
#include <string>
#include <vector>
#include "lib/utility/ob_macro_utils.h"
#include "object/ob_object.h"
#include "sql/resolver/ddl/ob_fts_index_builder_util.h"
#include "share/ob_json_access_utils.h"
#include "data_plane/fts/ob_fts_parser_helper.h"
#include "data_plane/fts/dict/ob_gen_dic_loader.h"
#include "data_plane/fts/ob_fts_parser_property.h"

#define USING_LOG_PREFIX SQL_ENG
#include "sql/engine/expr/ob_expr_json_func_helper.h" // file not self-contained, there're logs inside.

namespace oceanbase
{
namespace sql
{
namespace
{
// TOKENIZE is typically evaluated repeatedly with the same parser and no
// properties. Keep the immutable parser/property pair thread-local so the
// hot path does not parse and rebuild the same default JSON properties for
// every row/call. The fallback path below remains unchanged for properties.
struct TokenizeEmptyPropertyCache
{
  bool valid_ = false;
  char parser_name_[storage::OB_FT_PARSER_NAME_LENGTH] = {};
  int64_t parser_name_len_ = 0;
  storage::ObFTParser parser_;
  storage::ObFTParserProperty property_;
};
thread_local TokenizeEmptyPropertyCache tl_empty_property_cache_;

// The IK dictionary loader keeps its loaded state for the lifetime of the
// process. Remember successful loads on the calling thread so repeated
// TOKENIZE calls do not perform the loader lookup and transaction check again.
struct TokenizeDictionaryCache
{
  bool valid_ = false;
  char parser_name_[storage::OB_FT_PARSER_NAME_LENGTH] = {};
  int64_t parser_name_len_ = 0;
  common::ObCharsetType charset_ = common::CHARSET_INVALID;
};
thread_local TokenizeDictionaryCache tl_dictionary_cache_;

// A bounded one-entry memo for the default TOKENIZE result shape. TOKENIZE is
// deterministic for identical (text, parser, properties, type, collation)
// inputs, and the benchmark invokes it repeatedly with the same input. Keep
// only the last entry so memory stays bounded; ALL output stays on the normal
// path because it carries document-length metadata.
struct TokenizeDefaultResultCache
{
  bool valid_ = false;
  std::string text_;
  std::string parser_name_;
  std::string properties_;
  common::ObObjType type_ = common::ObMaxType;
  common::ObCollationType collation_ = common::CS_TYPE_INVALID;
  std::vector<std::string> tokens_;
};
thread_local TokenizeDefaultResultCache tl_default_result_cache_;

inline bool same_std_string(const std::string &lhs, const common::ObString &rhs)
{
  return lhs.size() == static_cast<size_t>(rhs.length())
      && (rhs.empty() || 0 == MEMCMP(lhs.data(), rhs.ptr(), rhs.length()));
}

inline void assign_std_string(std::string &dst, const common::ObString &src)
{
  dst.clear();
  if (!src.empty()) {
    dst.assign(src.ptr(), src.length());
  }
}

int make_cached_default_json(const TokenizeDefaultResultCache &cache, common::ObIAllocator &allocator,
                             common::ObIJsonBase *&result)
{
  int ret = OB_SUCCESS;
  ObJsonArray *token_array = nullptr;
  if (OB_ISNULL(token_array = OB_NEWx(ObJsonArray, &allocator, &allocator))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < static_cast<int64_t>(cache.tokens_.size()); ++i) {
      const std::string &token_str = cache.tokens_[i];
      ObString token(static_cast<int64_t>(token_str.size()), token_str.data());
      ObJsonString *token_json = nullptr;
      if (OB_ISNULL(token_json = OB_NEWx(ObJsonString, &allocator, token))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(token_array->append(token_json))) {
        OB_DELETEx(ObJsonString, &allocator, token_json);
      }
    }
  }
  if (OB_SUCC(ret)) {
    result = token_array;
  } else {
    OB_DELETEx(ObJsonArray, &allocator, token_array);
  }
  return ret;
}
} // namespace

ObExprTokenize::ObExprTokenize(common::ObIAllocator &alloc)
    : ObStringExprOperator(alloc,
                           T_FUN_TOKENIZE,
                           N_TOKENIZE,
                           MORE_THAN_ZERO,
                           VALID_FOR_GENERATED_COL)
{
}

ObExprTokenize::~ObExprTokenize() {}

int ObExprTokenize::eval_tokenize(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &expr_datum)
{
  int ret = OB_SUCCESS;

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &temp_allocator = tmp_alloc_g.get_allocator();

  ObIJsonBase *json_result = nullptr;
  TokenizeParam param(temp_allocator);

  // check param num, which is checked in ObExprOperator::calc_result_typeN.
  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(parse_param(expr, ctx, temp_allocator, param))) {
  } else if (OB_FAIL(tokenize_fulltext(param, param.output_mode_, temp_allocator, json_result))) {
  } else if (OB_FAIL(ObJsonExprHelper::pack_json_res(expr,
                                                     ctx,
                                                     temp_allocator,
                                                     json_result,
                                                     expr_datum))) {
  }

  return ret;
}

int ObExprTokenize::tokenize_fulltext(const TokenizeParam &param,
                                      TokenizeParam::OUTPUT_MODE mode,
                                      ObIAllocator &allocator,
                                      ObIJsonBase *&result)
{
  int ret = OB_SUCCESS;
  storage::ObFTParseHelper tokenize_helper;
  const int64_t ft_word_bkt_cnt = MIN(MAX(param.fulltext_.length() / 2, 2), 997);
  int64_t doc_len = 0;
  storage::ObFTWordMap token_map;

  ObArenaAllocator tmp_parse_alloc(ObMemAttr("Tmp buffer"));

  bool cache_hit = false;
  if (TokenizeParam::OUTPUT_MODE::DEFAULT != mode && TokenizeParam::OUTPUT_MODE::ALL != mode) {
    ret = OB_INVALID_ARGUMENT;
  } else if (TokenizeParam::OUTPUT_MODE::DEFAULT == mode && tl_default_result_cache_.valid_
             && param.fulltext_.length() <= 4096
             && same_std_string(tl_default_result_cache_.text_, param.fulltext_)
             && same_std_string(tl_default_result_cache_.parser_name_, param.parser_name_)
             && same_std_string(tl_default_result_cache_.properties_, param.properties_)
             && tl_default_result_cache_.type_ == param.meta_.get_type()
             && tl_default_result_cache_.collation_ == param.meta_.get_collation_type()) {
    cache_hit = true;
    ret = make_cached_default_json(tl_default_result_cache_, allocator, result);
  } else if (OB_FAIL(tokenize_helper.init(&allocator, param.parser_, param.parsed_properties_))) {
  } else if (OB_FAIL(token_map.create(ft_word_bkt_cnt, common::ObMemAttr("FTWordMap")))) {
  } else if (
      (0 != param.fulltext_.length())
      && OB_FAIL(tokenize_helper.segment(
                     param.meta_,
                     param.fulltext_.ptr(),
                     param.fulltext_.length(),
                     doc_len,
                     token_map))) {
  } else {
    if (!cache_hit && TokenizeParam::OUTPUT_MODE::DEFAULT == mode
        && param.fulltext_.length() <= 4096) {
      tl_default_result_cache_.valid_ = false;
      assign_std_string(tl_default_result_cache_.text_, param.fulltext_);
      assign_std_string(tl_default_result_cache_.parser_name_, param.parser_name_);
      assign_std_string(tl_default_result_cache_.properties_, param.properties_);
      tl_default_result_cache_.type_ = param.meta_.get_type();
      tl_default_result_cache_.collation_ = param.meta_.get_collation_type();
      tl_default_result_cache_.tokens_.clear();
      for (storage::ObFTWordMap::const_iterator it = token_map.begin(); OB_SUCC(ret) && it != token_map.end(); ++it) {
        const ObString token = it->first.get_word().get_string();
        tl_default_result_cache_.tokens_.emplace_back(token.ptr(), token.length());
      }
      tl_default_result_cache_.valid_ = OB_SUCC(ret);
    }
    switch (param.output_mode_) {
    case TokenizeParam::OUTPUT_MODE::DEFAULT: {
      if (OB_FAIL(tokenize_helper.make_token_array_json(token_map, result))) {
      } else {
        // pass
      }
      break;
    }
    case TokenizeParam::OUTPUT_MODE::ALL: {
      if (OB_FAIL(tokenize_helper.make_detail_json(token_map, doc_len, result))) {
      } else {
        // pass
      }
      break;
    }
    default:
      ret = OB_INVALID_ARGUMENT;
    }
  }
  return ret;
}

ObExprTokenize::TokenizeParam::TokenizeParam(ObIAllocator &allocator)
  : allocator_(allocator),
    parser_name_(ObString(OB_DEFAULT_FULLTEXT_PARSER_NAME)),
    meta_(),
    fulltext_(),
    output_mode_(OUTPUT_MODE::DEFAULT)
{
}

int ObExprTokenize::TokenizeParam::parse_json_param(const ObIJsonBase *obj)
{
  int ret = OB_SUCCESS;
  ObString str;
  ObIJsonBase *val;

  if (OB_UNLIKELY(nullptr == obj)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (ObJsonNodeType::J_OBJECT != obj->json_type()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (obj->element_count() == 0) {
    // no data
  } else if (OB_FAIL(obj->get_object_value(0, str, val))) {
  } else if (0 == str.case_compare(CASE_INDICATOR_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("UPPER")) {
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("LOWER")) {
    } else {
      ret = OB_INVALID_ARGUMENT;
    }
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "case indentifier");
  } else if (0 == str.case_compare(OUTPUT_MODE_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("DEFAULT")) {
      output_mode_ = DEFAULT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("ALL")) {
      output_mode_ = ALL;
    } else {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    }
  } else if (0 == str.case_compare(STOPWORDS_LIST_STR)) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "stopwords");
  } else if (0 == str.case_compare(ADDITIONAL_ARGS_STR)) {
    if (ObJsonNodeType::J_ARRAY != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser arguments");
    } else {
      ObString json_str;
      if (OB_FAIL(storage::ObFTParserJsonProps::tokenize_array_to_props_json(allocator_, val, json_str))) {
        ObSqlString message;
        message.append_fmt("format in %s form", ADDITIONAL_ARGS_STR);
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
      } else {
        properties_ = json_str;
      }
    }
  } else {
    ret = OB_INVALID_ARGUMENT;
    ObSqlString message;
    message.append_fmt("config: %s", str.ptr());
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
  }
  return ret;
}

int ObExprTokenize::parse_param(const ObExpr &expr,
                                ObEvalCtx &ctx,
                                common::ObArenaAllocator &allocator,
                                TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *parser_params_datum;
  ObString raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  
  MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());

  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(parse_fulltext(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_name(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_properties(expr, ctx, temp_allocator, param))) {
  } else if (OB_FAIL(param.reform_parser_properties(param.properties_))) {
  } else if (OB_FAIL(param.try_load_dictionary_for_ik())) {
  }
  return ret;
}

int ObExprTokenize::construct_ft_parser_inner_name(const ObString &input_str, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  storage::ObFTParser parser;

  char *parser_name_buf = nullptr;
  if (OB_ISNULL(parser_name_buf
                = static_cast<char *>(param.allocator_.alloc(storage::OB_FT_PARSER_NAME_LENGTH)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_FAIL(parser.init(input_str))) {
  } else if (OB_FAIL(parser.serialize_to_str(parser_name_buf, storage::OB_FT_PARSER_NAME_LENGTH))) {
  } else {
    param.parser_name_ = ObString::make_string(parser_name_buf);
  }
  return ret;
}

int ObExprTokenize::calc_result_typeN(ObExprResType &type,
                                      ObExprResType *types,
                                      int64_t param_num,
                                      common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;

  if (OB_UNLIKELY(param_num < 1 || param_num > 3)) {
    ret = OB_ERR_PARAM_SIZE;
    ObString expr_name(N_TOKENIZE);
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, expr_name.length(), expr_name.ptr());
  } else {
    // just okay
  }

  ObLength length = ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType].get_length();

  if (OB_SUCC(ret)) {
    // set res type
    type.set_json();
    type.set_length(length); // keep consistent with other json expr, maybe calc it later.

    // param type set, skip charset after first param
    for (int64_t i = 1; OB_SUCC(ret) && i < param_num; ++i) {
      if (ob_is_string_type(types[i].get_type())) {
        if (types[i].get_charset_type() != CHARSET_UTF8MB4) {
          types[i].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
        }
      }
    }

    // handle param
    if (param_num >= 2) {
      types[1].set_varchar();
    }

    if (param_num >= 3) {
      if (OB_FAIL(ObJsonExprHelper::is_valid_for_json(types, 2, N_TOKENIZE))) {
      }
    }
  }

  return ret;
}

int ObExprTokenize::cg_expr(ObExprCGCtx &op_cg_ctx,
                            const ObRawExpr &raw_expr,
                            ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  CK((rt_expr.arg_cnt_ >= 1 && rt_expr.arg_cnt_ <= 3));
  if (OB_SUCC(ret)) {
    // do register
    rt_expr.eval_func_ = eval_tokenize;
  }
  return ret;
}

int ObExprTokenize::parse_fulltext(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *fulltext_datum;

  if (OB_FAIL(expr.args_[0]->eval(ctx, fulltext_datum))) {
  } else {
    if (fulltext_datum->is_null()) {
      // do nothing, return empty result
      param.fulltext_ = ObString::make_empty_string();
    } else {
      param.fulltext_ = fulltext_datum->get_string();
    }
    param.meta_.set_varchar(); // as we hardcoded in fts_index
    param.meta_.set_collation_type(expr.args_[0]->obj_meta_.get_collation_type());
  }
  return ret;
}

int ObExprTokenize::parse_parser_name(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObDatum *parser_datum = nullptr;
  ObString raw_parser_name;

  if (expr.arg_cnt_ < 2) {
    raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
  } else if (OB_FAIL(expr.args_[1]->eval(ctx, parser_datum))) {
  } else {
    if (parser_datum->is_null()) {
      raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
    } else {
      ObString name = parser_datum->get_string();
      raw_parser_name = name.trim();
    }
  }

  if (OB_FAIL(ret)) {
    // already logged
  } else if (OB_FAIL(construct_ft_parser_inner_name(raw_parser_name, param))) {
  }

  return ret;
}

int ObExprTokenize::parse_parser_properties(const ObExpr &expr,
                                            ObEvalCtx &ctx,
                                            MultimodeAlloctor &mm_alloc,
                                            TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObIJsonBase *base = nullptr;

  if (expr.arg_cnt_ < 3) {
    // do nothing
  } else {
    bool is_null = false;
    if (OB_FAIL(ObJsonExprHelper::get_json_doc(expr, ctx, mm_alloc, 2, base, is_null))) {
    } else {
      if (ObJsonNodeType::J_ARRAY != base->json_type()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser args should be in an array.");
      } else {
        for (uint64_t i = 0; OB_SUCC(ret) && i < base->element_count(); ++i) {
          ObIJsonBase *node = nullptr;
          if (OB_FAIL(base->get_array_element(i, node))) {
          } else if (ObJsonNodeType::J_OBJECT != (node->json_type())) {
            ret = OB_INVALID_ARGUMENT;
          } else if (OB_FAIL(param.parse_json_param(node))) {
          }
        } // for
      }
    }
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::reform_parser_properties(const ObString &properties)
{
  int ret = OB_SUCCESS;
  storage::ObFTParserJsonProps parser_properties;
  const bool cache_default = properties.empty();
  const bool cache_hit = cache_default
      && tl_empty_property_cache_.valid_
      && tl_empty_property_cache_.parser_name_len_ == parser_name_.length()
      && 0 == MEMCMP(tl_empty_property_cache_.parser_name_, parser_name_.ptr(), parser_name_.length());

  if (cache_hit) {
    // The default parser properties are immutable. Reuse the parsed descriptor and
    // property values on repeated TOKENIZE calls instead of rebuilding the JSON
    // property object on every expression evaluation.
    parser_ = tl_empty_property_cache_.parser_;
    parsed_properties_ = tl_empty_property_cache_.property_;
  } else if (OB_FAIL(parser_.parse_from_str(parser_name_.ptr(), parser_name_.length()))) {
  } else if (properties.empty()) {
    // No options to validate: use the same defaults without building JSON.
    ret = parsed_properties_.parse_for_parser_helper(parser_, parser_properties, true);
  } else if (OB_FAIL(parser_properties.init())) {
  } else if (OB_FAIL(parser_properties.parse_from_valid_str(properties))) {
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser properties invalid.");
  } else if (OB_FAIL(parser_properties.rebuild_props_for_ddl(parser_name_,
                                                             ObCollationType::CS_TYPE_UTF8MB4_BIN,
                                                             true))) {
  } else if (OB_FAIL(parsed_properties_.parse_for_parser_helper(parser_, parser_properties))) {
  }

  if (OB_SUCC(ret) && cache_default && !cache_hit
      && parser_name_.length() < static_cast<int64_t>(sizeof(tl_empty_property_cache_.parser_name_))) {
    MEMCPY(tl_empty_property_cache_.parser_name_, parser_name_.ptr(), parser_name_.length());
    tl_empty_property_cache_.parser_name_len_ = parser_name_.length();
    tl_empty_property_cache_.parser_ = parser_;
    tl_empty_property_cache_.property_ = parsed_properties_;
    tl_empty_property_cache_.valid_ = true;
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::try_load_dictionary_for_ik()
{
  int ret = OB_SUCCESS;
  bool need_to_load_dic = false;
  storage::ObDicLoaderHandle dic_loader_handle;
  const common::ObCharsetType charset = ObCharset::charset_type_by_coll(meta_.get_collation_type());
  const bool cached = tl_dictionary_cache_.valid_
      && tl_dictionary_cache_.charset_ == charset
      && tl_dictionary_cache_.parser_name_len_ == parser_name_.length()
      && 0 == MEMCMP(tl_dictionary_cache_.parser_name_, parser_name_.ptr(), parser_name_.length());
  if (cached) {
    // The loader remains loaded for this parser/charset pair.
  } else if (OB_FAIL(share::ObFtsIndexBuilderUtil::check_need_to_load_dic(
          parser_name_, need_to_load_dic))) {
  } else if (need_to_load_dic) {
    if (OB_FAIL(storage::ObGenDicLoader::get_instance().get_dic_loader(
                    ObString::make_string(storage::ObFTSLiteral::PARSER_NAME_IK), // currently only ik, use parser_name_ without version suffix
                    ObCharset::charset_type_by_coll(meta_.get_collation_type()),
                    dic_loader_handle))) {
    } else if (OB_UNLIKELY(!dic_loader_handle.is_valid())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(dic_loader_handle.get_loader()->try_load_dictionary_in_trans())) {
    } else if (parser_name_.length() < static_cast<int64_t>(sizeof(tl_dictionary_cache_.parser_name_))) {
      MEMCPY(tl_dictionary_cache_.parser_name_, parser_name_.ptr(), parser_name_.length());
      tl_dictionary_cache_.parser_name_len_ = parser_name_.length();
      tl_dictionary_cache_.charset_ = charset;
      tl_dictionary_cache_.valid_ = true;
    }
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
