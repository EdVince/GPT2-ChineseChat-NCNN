// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef GPT2_H
#define GPT2_H

#include <map>
#include <net.h>
#include <vector>

class GPT2
{
public:
    GPT2();

    int load(AAssetManager* mgr, std::string vocab);
    std::string chat(std::string in);

private:
    std::vector<int> token2idx(std::string token);
    std::string idx2token(std::vector<int> idx);

private:
    ncnn::Net net;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;

    std::map<std::wstring, int> tokenizer_token2idx;
    std::map<int, std::wstring> tokenizer_idx2token;

    const int max_history_len = 3;
    const int max_len = 25;

    std::vector<std::vector<int>> history;
};

#endif // NANODET_H
