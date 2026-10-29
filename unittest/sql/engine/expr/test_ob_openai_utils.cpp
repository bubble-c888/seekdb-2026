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

#include <gtest/gtest.h>
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"
#include "common/json_type/ob_json_base.h"

namespace oceanbase
{
namespace common
{

class ObOpenAIUtilsTest : public ::testing::Test
{
public:
  virtual void SetUp() {}
  virtual void TearDown() {}

protected:
  ObArenaAllocator allocator_;

  int build_json(const char *text, ObJsonObject *&obj)
  {
    ObIJsonBase *j_base = nullptr;
    ObString json_text(text);
    int ret = ObJsonBaseFactory::get_json_base(
        &allocator_, json_text, ObJsonInType::JSON_TREE, ObJsonInType::JSON_TREE, j_base);
    if (OB_SUCCESS == ret) {
      obj = static_cast<ObJsonObject *>(j_base);
    }
    return ret;
  }
};

TEST_F(ObOpenAIUtilsTest, parse_complete_output_valid)
{
  ObOpenAIUtils::ObOpenAIComplete provider;
  ObJsonObject *response = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_json(
      "{\"choices\": [{\"message\": {\"role\": \"assistant\", \"content\": \"hello world\"}}]}",
      response));
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, response, result));
  ASSERT_NE(nullptr, result);
  EXPECT_EQ(ObJsonNodeType::J_STRING, result->json_type());
  EXPECT_STREQ("hello world", static_cast<ObJsonString *>(result)->get_str().ptr());
}

TEST_F(ObOpenAIUtilsTest, parse_complete_output_empty_content)
{
  ObOpenAIUtils::ObOpenAIComplete provider;
  ObJsonObject *response = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_json(
      "{\"choices\": [{\"message\": {\"content\": \"\"}}]}",
      response));
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_INVALID_DATA, provider.parse_output(allocator_, response, result));
}

TEST_F(ObOpenAIUtilsTest, parse_complete_output_non_string_content)
{
  ObOpenAIUtils::ObOpenAIComplete provider;
  ObJsonObject *response = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_json(
      "{\"choices\": [{\"message\": {\"content\": 123}}]}",
      response));
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_INVALID_DATA, provider.parse_output(allocator_, response, result));
}

TEST_F(ObOpenAIUtilsTest, parse_complete_output_null_response)
{
  ObOpenAIUtils::ObOpenAIComplete provider;
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_INVALID_ARGUMENT, provider.parse_output(allocator_, nullptr, result));
}

TEST_F(ObOpenAIUtilsTest, parse_ollama_complete_output_valid)
{
  ObOllamaUtils::ObOllamaComplete provider;
  ObJsonObject *response = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_json(
      "{\"model\": \"llama3.1\", \"response\": \"ok answer\", \"done\": true}",
      response));
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, response, result));
  ASSERT_NE(nullptr, result);
  EXPECT_EQ(ObJsonNodeType::J_STRING, result->json_type());
  EXPECT_STREQ("ok answer", static_cast<ObJsonString *>(result)->get_str().ptr());
}

TEST_F(ObOpenAIUtilsTest, parse_ollama_complete_output_missing_response)
{
  ObOllamaUtils::ObOllamaComplete provider;
  ObJsonObject *response = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_json("{\"model\": \"llama3.1\", \"done\": true}", response));
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_ERR_UNEXPECTED, provider.parse_output(allocator_, response, result));
}

TEST_F(ObOpenAIUtilsTest, parse_embed_output_out_of_order_index)
{
  // response items arrive out of order; parse must restore input order by index
  ObOpenAIUtils::ObOpenAIEmbed provider;
  ObJsonObject *response = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_json(
      "{\"data\": ["
      "{\"object\": \"embedding\", \"index\": 1, \"embedding\": [0.3, 0.4]},"
      "{\"object\": \"embedding\", \"index\": 0, \"embedding\": [0.1, 0.2]}"
      "]}",
      response));
  ObIJsonBase *result = nullptr;
  ASSERT_EQ(OB_SUCCESS, provider.parse_output(allocator_, response, result));
  ASSERT_NE(nullptr, result);
  ASSERT_EQ(ObJsonNodeType::J_ARRAY, result->json_type());
  ObJsonArray *result_array = static_cast<ObJsonArray *>(result);
  ASSERT_EQ(2, result_array->element_count());
  ObIJsonBase *first = result_array->get_value(0);
  ASSERT_NE(nullptr, first);
  EXPECT_EQ(ObJsonNodeType::J_ARRAY, first->json_type());
  ObJsonNode *first_elem = static_cast<ObJsonArray *>(first)->get_value(0);
  ASSERT_NE(nullptr, first_elem);
  EXPECT_DOUBLE_EQ(0.1, static_cast<ObJsonDouble *>(first_elem)->value());
  ObJsonNode *second_elem = static_cast<ObJsonArray *>(result_array->get_value(1))->get_value(0);
  ASSERT_NE(nullptr, second_elem);
  EXPECT_DOUBLE_EQ(0.3, static_cast<ObJsonDouble *>(second_elem)->value());
}

TEST_F(ObOpenAIUtilsTest, parse_embed_output_duplicate_index_fails)
{
  ObOpenAIUtils::ObOpenAIEmbed provider;
  ObJsonObject *response = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_json(
      "{\"data\": ["
      "{\"object\": \"embedding\", \"index\": 0, \"embedding\": [0.1]},"
      "{\"object\": \"embedding\", \"index\": 0, \"embedding\": [0.2]}"
      "]}",
      response));
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_INVALID_DATA, provider.parse_output(allocator_, response, result));
}

TEST_F(ObOpenAIUtilsTest, parse_embed_output_index_out_of_range_fails)
{
  ObOpenAIUtils::ObOpenAIEmbed provider;
  ObJsonObject *response = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_json(
      "{\"data\": ["
      "{\"object\": \"embedding\", \"index\": 5, \"embedding\": [0.1]}"
      "]}",
      response));
  ObIJsonBase *result = nullptr;
  EXPECT_EQ(OB_INVALID_DATA, provider.parse_output(allocator_, response, result));
}

} // namespace common
} // namespace oceanbase

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
