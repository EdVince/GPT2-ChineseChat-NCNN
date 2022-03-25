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

#include "gpt2.h"
#include <map>
#include <fstream>
#include <string>
#include <cstdlib>
#include <wchar.h>
#include <iostream>
#include <codecvt>
#include <ctime>
#include <algorithm>
#include <functional>
#include <numeric>
#include <time.h>

#include "cpu.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO , "GPT2", __VA_ARGS__)

int __Neg_Infinity = 0xFF800000;
const float Neg_Infinity = *((float*)&__Neg_Infinity);

class DivTrilWhere : public ncnn::Layer
{
public:
    DivTrilWhere()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        top_blob.create(w, h, channels, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const float* src = bottom_blob.channel(p);
            float* dst = top_blob.channel(p);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    if (x > y) {
                        dst[0] = -1e4f;
                    }
                    else {
                        dst[0] = src[0] / 8.0f;
                    }
                    src++;
                    dst++;
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(DivTrilWhere)

class Gather : public ncnn::Layer
{
public:
    Gather()
    {
        one_blob_only = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        int w = bottom_blobs[1].w;
        int vocab_size = bottom_blobs[0].h;
        int n_embd = bottom_blobs[0].w;

        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(n_embd, w, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float* dst = top_blob;
        const float* in = bottom_blobs[1];
        const float* weight = bottom_blobs[0];

#pragma omp parallel for num_threads(opt.num_threads)
        for (int c = 0; c < w; c++) {
            int idx = std::round(*in) * n_embd;
            memcpy(dst, weight + idx, n_embd * 4);
            in++;
            dst += n_embd;
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Gather)

std::string WStringToString(const std::wstring& wstr)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;
    return converterX.to_bytes(wstr);
}

std::wstring StringToWString(const std::string& str)
{
    using convert_typeX = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_typeX, wchar_t> converterX;
    return converterX.from_bytes(str);
}

std::vector<int> vector_merge(std::vector<int> v1, std::vector<int> v2)
{
    std::vector<int> v3;
    v3.insert(v3.end(), v1.begin(), v1.end());
    v3.insert(v3.end(), v2.begin(), v2.end());
    return v3;
}

void top_k_filtering(ncnn::Mat& logits)
{
    ncnn::Mat filtered_logits;
    filtered_logits.clone_from(logits);
    float* pt = filtered_logits;
    std::sort(pt, pt + 13317, std::greater<float>());
    float top_k_value = pt[8 - 1]; // topk的阈值
    for (int i = 0; i < 13317; i++) {
        if (logits[i] < top_k_value)
            logits[i] = Neg_Infinity;
    }
}

template<typename _Tp>
int softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = std::exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

int multinomial(const ncnn::Mat& logits)
{
    ncnn::Mat weight;
    weight.clone_from(logits);
    for (int i = 1; i < 13317; i++)
        weight[i] += weight[i - 1];
    std::srand(static_cast <unsigned> (time(NULL)));
    float r = static_cast<float>(rand() % 13317) / 13317.0f;
    float* pt = weight;
    return std::lower_bound(pt, pt + 13317, r) - pt;
}

GPT2::GPT2()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int GPT2::load(AAssetManager* mgr, std::string vocab)
{
    net.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net.opt = ncnn::Option();
#if NCNN_VULKAN
    net.opt.use_vulkan_compute = 0;
#endif
    net.opt.lightmode = true;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.blob_allocator = &blob_pool_allocator;
    net.opt.workspace_allocator = &workspace_pool_allocator;

    net.register_custom_layer("DivTrilWhere", DivTrilWhere_layer_creator);
    net.register_custom_layer("Gather", Gather_layer_creator);
    net.load_param(mgr, "gpt2.param");
    net.load_model(mgr, "gpt2.bin");

    LOGI("load ncnn model ok!");


    std::ifstream infile;
    infile.open(vocab.data());
    std::string s;
    int idx = 0;
    while (getline(infile, s)) {
        auto ws = StringToWString(s);
        tokenizer_token2idx.insert(std::pair<std::wstring, int>(ws, idx));
        tokenizer_idx2token.insert(std::pair<int, std::wstring>(idx, ws));
        idx++;
    }
    infile.close();

    LOGI("load vocab: %d\n", idx);


    return 0;
}

std::vector<int> GPT2::token2idx(std::string token)
{
    std::vector<int> idx;
    std::wstring wtoken = StringToWString(token);
    for(int i = 0; i < wtoken.length(); i++) {
        std::wstring tmp = wtoken.substr(i,1);
        idx.push_back(tokenizer_token2idx[tmp]);
    }
    return idx;
}

std::string GPT2::idx2token(std::vector<int> idx)
{
    std::wstring wtoken;
    for(int i = 0; i < idx.size(); i++){
        wtoken += tokenizer_idx2token[idx[i]];
    }
    std::string token = WStringToString(wtoken);
    return token;
}

std::string GPT2::chat(std::string in)
{
    std::vector<int> text_ids = token2idx(in);
    history.push_back(text_ids);
    std::vector<int> input_ids = { 101 };
    int history_len = 3;
    if (history.size() < max_history_len)
        history_len = history.size();
    std::vector<std::vector<int>> max_history;
    max_history.assign(history.end() - history_len, history.end());
    for (std::vector<int> history_utr : max_history) {
        input_ids = vector_merge(input_ids, history_utr);
        input_ids.push_back(102);
    }

    std::vector<int> response;
    for (int it = 0; it < max_len; it++) {

        ncnn::Mat input_ids_mat(input_ids.size());
        ncnn::Mat position_ids_mat(input_ids.size());
        for (int i = 0; i < input_ids.size(); i++) {
            input_ids_mat[i] = float(input_ids[i]);
            position_ids_mat[i] = float(i);
        }

        ncnn::Mat logits;
        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("0", input_ids_mat);
            ex.input("input.3", position_ids_mat);
            ex.extract("1673", logits);
        }

        ncnn::Mat next_token_logits;
        next_token_logits.clone_from(logits.row_range(logits.h - 1, 1));
        next_token_logits[100] = Neg_Infinity;
        top_k_filtering(next_token_logits);
        softmax<float>(next_token_logits, next_token_logits, 13317);
        int next_token = multinomial(next_token_logits);
        if (next_token == 102) break;
        response.push_back(next_token);
        input_ids.push_back(next_token);
    }

    history.push_back(response);
    std::string bot_text = idx2token(response);

    return bot_text;
}
