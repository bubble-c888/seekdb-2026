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

#include "ob_super_block_struct.h"
#include "storage/blocksstable/ob_block_sstable_struct.h"

namespace oceanbase
{
namespace storage
{

using namespace oceanbase::common;
using namespace oceanbase::blocksstable;
using namespace oceanbase::share;

const MacroBlockId ObServerSuperBlock::EMPTY_LIST_ENTRY_BLOCK(0, MacroBlockId::EMPTY_ENTRY_BLOCK_INDEX, 0);

// ========================== ObServerSuperBlock ==============================

ObServerSuperBlockHeader::ObServerSuperBlockHeader()
{
  reset();
}

bool ObServerSuperBlockHeader::is_valid() const
{
  return version_ == SERVER_SUPER_BLOCK_VERSION
      && magic_ == SERVER_SUPER_BLOCK_MAGIC && body_size_ > 0;
}

void ObServerSuperBlockHeader::reset()
{
  version_ = 0;
  magic_ = 0;
  body_size_ = 0;
  body_crc_ = 0;
}

DEFINE_SERIALIZE(ObServerSuperBlockHeader)
{
  int ret = OB_SUCCESS;
  if (NULL == buf || buf_len - pos < sizeof(ObServerSuperBlockHeader)) {
    ret = OB_BUF_NOT_ENOUGH;
  } else {
    MEMCPY(buf + pos, this, sizeof(ObServerSuperBlockHeader));
    pos += sizeof(ObServerSuperBlockHeader);
  }
  return ret;
}

DEFINE_DESERIALIZE(ObServerSuperBlockHeader)
{
  int ret = OB_SUCCESS;
  if (NULL == buf || data_len - pos < sizeof(ObServerSuperBlockHeader)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    MEMCPY(this, buf + pos, sizeof(ObServerSuperBlockHeader));
    pos += sizeof(ObServerSuperBlockHeader);
  }

  if (OB_SUCC(ret)) {
    if (OB_UNLIKELY(SERVER_SUPER_BLOCK_VERSION != version_ || SERVER_SUPER_BLOCK_MAGIC != magic_)) {
      ret = OB_ERR_SYS;
      LOG_ERROR("version or magic not match", K(ret), K(*this));
    }
  }
  return ret;
}

DEFINE_GET_SERIALIZE_SIZE(ObServerSuperBlockHeader)
{
  return sizeof(ObServerSuperBlockHeader);
}

ServerSuperBlockBody::ServerSuperBlockBody()
{
  reset();
}

bool ServerSuperBlockBody::is_valid() const
{
  return create_timestamp_ > 0 && modify_timestamp_ >= create_timestamp_ &&
    macro_block_size_ > 0 && total_macro_block_count_ > 0 &&
    total_file_size_ >= macro_block_size_;
}

void ServerSuperBlockBody::reset()
{
  create_timestamp_ = 0;
  modify_timestamp_ = 0;
  macro_block_size_ = 0;
  total_macro_block_count_ = 0;
  total_file_size_ = 0;
}

OB_UNIS_SERIALIZE(ServerSuperBlockBody);                                         
OB_UNIS_DESERIALIZE(ServerSuperBlockBody);          
OB_UNIS_SERIALIZE_SIZE(ServerSuperBlockBody);           
int ServerSuperBlockBody::serialize_(char *buf, const int64_t buf_len, int64_t &pos) const
{
  int ret = OB_SUCCESS;
  LST_DO_CODE(OB_UNIS_ENCODE, create_timestamp_,
      modify_timestamp_, macro_block_size_, total_macro_block_count_, total_file_size_);
  return ret;
}
int ServerSuperBlockBody::deserialize_(const char *buf, const int64_t data_len, int64_t &pos)
{
  int ret = OB_SUCCESS;
  LST_DO_CODE(OB_UNIS_DECODE, create_timestamp_,
      modify_timestamp_, macro_block_size_, total_macro_block_count_, total_file_size_);
  return ret;
}
int64_t ServerSuperBlockBody::get_serialize_size_(void) const
{
  int64_t len = 0;
  LST_DO_CODE(OB_UNIS_ADD_LEN, create_timestamp_,
      modify_timestamp_, macro_block_size_, total_macro_block_count_, total_file_size_);
  return len;
}

ObServerSuperBlock::ObServerSuperBlock() : header_(), body_() {}

bool ObServerSuperBlock::is_valid() const
{
  return header_.is_valid() && body_.is_valid();
}

int64_t ObServerSuperBlock::get_serialize_size() const
{
  return header_.get_serialize_size() + body_.get_serialize_size();
}

int ObServerSuperBlock::serialize(char *buf, const int64_t buf_size, int64_t &pos) const
{
  int ret = OB_SUCCESS;
  int64_t new_pos = pos;

  if (OB_ISNULL(buf) || buf_size <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else if (!is_valid()) {
    ret = OB_ERR_SYS;
  } else {
    MEMSET(buf + pos, 0, buf_size - pos);
    if (OB_FAIL(header_.serialize(buf, buf_size, new_pos))) {
    } else if (OB_FAIL(body_.serialize(buf, buf_size, new_pos))) {
    } else {
      pos = new_pos;
      LOG_INFO("succeed to serialize super block buf", K(buf_size), K(pos), K(*this));
    }
  }
  return ret;
}

int ObServerSuperBlock::deserialize(const char *buf, const int64_t buf_size, int64_t &pos)
{
  int ret = OB_SUCCESS;
  int32_t calc_crc = 0;
  if (OB_ISNULL(buf) || buf_size <= 0 || pos < 0) {
    ret = OB_INVALID_ARGUMENT;
  } else if (is_valid()) {
    ret = OB_INIT_TWICE;
    LOG_ERROR("cannot read super block twice", K(ret), K(*this));
  } else if (OB_FAIL(header_.deserialize(buf, buf_size, pos))) {
  } else if (OB_UNLIKELY(header_.body_crc_ !=
      (calc_crc = static_cast<int32_t>(ob_crc64(buf + pos, header_.body_size_))))) {
    ret = OB_PHYSIC_CHECKSUM_ERROR;
    LOG_DBA_ERROR(OB_PHYSIC_CHECKSUM_ERROR, "msg", "failed to check crc", K(ret), KP(buf), K(buf_size), K(pos), K_(header), K(calc_crc));
  } else if (OB_FAIL(body_.deserialize(buf, buf_size, pos))) {
  } else if (OB_UNLIKELY(!is_valid())) {
    ret = OB_INVALID_DATA;
  } else {
    LOG_INFO("load server superblock success.", K(buf_size), K(pos), K(*this));
  }
  return ret;
}

void ObServerSuperBlock::reset()
{
  header_.reset();
  body_.reset();
}

int ObServerSuperBlock::construct_header()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!body_.is_valid())) {
    ret = OB_INVALID_DATA;
  } else {
    // calculate crc of content serialized buffer
    int64_t pos = 0;
    int64_t body_buf_len = body_.get_serialize_size();
    char *body_buf = static_cast<char *>(ob_malloc(body_buf_len, "SuperBlock"));
    if (OB_ISNULL(body_buf)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(body_.serialize(body_buf, body_buf_len, pos))) {
    } else {
      header_.version_ = ObServerSuperBlockHeader::SERVER_SUPER_BLOCK_VERSION;
      header_.magic_ = SERVER_SUPER_BLOCK_MAGIC;
      header_.body_size_ = body_buf_len;
      header_.body_crc_ = static_cast<int32_t>(ob_crc64(body_buf, body_buf_len));
    }
    if (OB_NOT_NULL(body_buf)) {
      ob_free(body_buf);
    }
  }
  return ret;
}

int ObServerSuperBlock::format_startup_super_block(
  const int64_t macro_block_size, const int64_t data_file_size)
{
  int ret = OB_SUCCESS;

  if (macro_block_size <= 0 || data_file_size <= 0 || data_file_size < macro_block_size) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    reset();
    body_.create_timestamp_ = ObTimeUtility::current_time();
    body_.modify_timestamp_ = body_.create_timestamp_;
    body_.macro_block_size_ = macro_block_size;
    body_.total_macro_block_count_ = data_file_size / macro_block_size;
    body_.total_file_size_ = lower_align(data_file_size, macro_block_size);

    if (OB_FAIL(construct_header())) {
    } else {
      LOG_INFO("success to format super block", K(*this));
    }
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase
