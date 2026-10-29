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
using namespace oceanbase::share;

class AIPromptUtilsTest : public ::testing::Test
{
public:
  virtual void SetUp() {}
  virtual void TearDown() {}

protected:
  ObArenaAllocator allocator_;
};

int build_prompt_object(ObIAllocator &allocator, const ObString &template_str,
                        const ObArray<ObString> &args, ObJsonObject *&prompt_object)
{
  int ret = OB_SUCCESS;
  ObJsonArray *args_array = nullptr;
  if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, args_array))) {
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < args.count(); ++i) {
      ObJsonString *arg_str = nullptr;
      ObString arg_val = args.at(i);
      if (OB_FAIL(ObAIFuncJsonUtils::get_json_string(allocator, arg_val, arg_str))) {
      } else if (OB_FAIL(args_array->append(arg_str))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    ObString tmpl = template_str;
    ret = ObAIFuncPromptObjectUtils::construct_prompt_object(allocator, tmpl, args_array, prompt_object);
  }
  return ret;
}

TEST_F(AIPromptUtilsTest, construct_prompt_object)
{
  int ret = OB_SUCCESS;
  ObString template_str("{0}+{1}={2}");
  ObArray<ObString> args;
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("1")));
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("1")));
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("2")));
  ObJsonObject *prompt_object = nullptr;
  ret = build_prompt_object(allocator_, template_str, args, prompt_object);
  ASSERT_EQ(OB_SUCCESS, ret);
  ASSERT_NE(nullptr, prompt_object);
  EXPECT_TRUE(ObAIFuncPromptObjectUtils::is_valid_prompt_object(prompt_object));
}

TEST_F(AIPromptUtilsTest, is_valid_prompt_object)
{
  EXPECT_FALSE(ObAIFuncPromptObjectUtils::is_valid_prompt_object(nullptr));

  ObIJsonBase *j_base = nullptr;
  ObString json_text("{\"template\": \"t{0}\", \"args\": [\"a\"]}");
  ASSERT_EQ(OB_SUCCESS, ObJsonBaseFactory::get_json_base(
      &allocator_, json_text, ObJsonInType::JSON_TREE, ObJsonInType::JSON_TREE, j_base));
  EXPECT_TRUE(ObAIFuncPromptObjectUtils::is_valid_prompt_object(
      static_cast<ObJsonObject *>(j_base)));

  ObIJsonBase *not_prompt = nullptr;
  ObString other_text("{\"foo\": \"bar\"}");
  ASSERT_EQ(OB_SUCCESS, ObJsonBaseFactory::get_json_base(
      &allocator_, other_text, ObJsonInType::JSON_TREE, ObJsonInType::JSON_TREE, not_prompt));
  EXPECT_FALSE(ObAIFuncPromptObjectUtils::is_valid_prompt_object(
      static_cast<ObJsonObject *>(not_prompt)));
}

TEST_F(AIPromptUtilsTest, replace_basic)
{
  ObString template_str("{0}+{1}={2}");
  ObArray<ObString> args;
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("1")));
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("1")));
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("2")));
  ObJsonObject *prompt_object = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_prompt_object(allocator_, template_str, args, prompt_object));
  ObString rendered;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, prompt_object, rendered));
  EXPECT_STREQ("1+1=2", rendered.ptr());
}

TEST_F(AIPromptUtilsTest, replace_duplicate_placeholders)
{
  ObString template_str("{0} and {0} and {1} {1} {1}");
  ObArray<ObString> args;
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("a")));
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("b")));
  ObJsonObject *prompt_object = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_prompt_object(allocator_, template_str, args, prompt_object));
  ObString rendered;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, prompt_object, rendered));
  EXPECT_STREQ("a and a and b b b", rendered.ptr());
}

TEST_F(AIPromptUtilsTest, replace_without_args)
{
  ObString template_str("no args here");
  ObArray<ObString> args;
  ObJsonObject *prompt_object = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_prompt_object(allocator_, template_str, args, prompt_object));
  ObString rendered;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, prompt_object, rendered));
  EXPECT_STREQ("no args here", rendered.ptr());
}

TEST_F(AIPromptUtilsTest, replace_non_digit_braces_copied_verbatim)
{
  ObString template_str("keep {x} and {} and {0a}");
  ObArray<ObString> args;
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("v")));
  ObJsonObject *prompt_object = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_prompt_object(allocator_, template_str, args, prompt_object));
  ObString rendered;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, prompt_object, rendered));
  EXPECT_STREQ("keep {x} and {} and {0a}", rendered.ptr());
}

TEST_F(AIPromptUtilsTest, replace_invalid_index_fails)
{
  ObString template_str("bad {3}");
  ObArray<ObString> args;
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString("a")));
  ObJsonObject *prompt_object = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_prompt_object(allocator_, template_str, args, prompt_object));
  ObString rendered;
  ASSERT_EQ(OB_INVALID_ARGUMENT, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, prompt_object, rendered));
}

TEST_F(AIPromptUtilsTest, replace_long_arg_content)
{
  ObString template_str("t={0}");
  ObArray<ObString> args;
  const char *long_str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  ASSERT_EQ(OB_SUCCESS, args.push_back(ObString(long_str)));
  ObJsonObject *prompt_object = nullptr;
  ASSERT_EQ(OB_SUCCESS, build_prompt_object(allocator_, template_str, args, prompt_object));
  ObString rendered;
  ASSERT_EQ(OB_SUCCESS, ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(
      allocator_, prompt_object, rendered));
  std::string expected = std::string("t=") + long_str;
  EXPECT_EQ(0, expected.compare(0, expected.size(), rendered.ptr(), rendered.length()));
}

} // namespace common
} // namespace oceanbase

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
