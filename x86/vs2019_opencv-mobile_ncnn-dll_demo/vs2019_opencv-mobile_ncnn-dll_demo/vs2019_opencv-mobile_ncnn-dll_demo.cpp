#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>
#include <fstream>
#include <map>
#include <limits>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <ctime>

#include "net.h"
#include "layer.h"


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


std::string WStringToString(const std::wstring& ws)
{
    std::string strLocale = setlocale(LC_ALL, "");
    const wchar_t* wchSrc = ws.c_str();
    size_t nDestSize = wcstombs(NULL, wchSrc, 0) + 1;
    char* chDest = new char[nDestSize];
    memset(chDest, 0, nDestSize);
    wcstombs(chDest, wchSrc, nDestSize);
    std::string strResult = chDest;
    delete[]chDest;
    setlocale(LC_ALL, strLocale.c_str());
    return strResult;
}

std::wstring StringToWString(const std::string& str)
{
    std::wstring wContext = L"";
    int len = MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.size(), NULL, 0);
    WCHAR* buffer = new WCHAR[len + 1];
    MultiByteToWideChar(CP_ACP, 0, str.c_str(), str.size(), buffer, len);
    buffer[len] = '\0';
    wContext.append(buffer);
    delete[] buffer;
    return wContext;
}

std::vector<std::string> SpiteStringCharacter(std::string context)
{
    std::vector<std::string> res;
    std::wstring wContext = StringToWString(context);
    for (int i = 0; i < wContext.length(); ++i) {
        std::wstring tmp = wContext.substr(i, 1);
        res.push_back(WStringToString(tmp));
    }
    return res;
}

std::string UTF8string(std::string strTemp)
{
    char buf[1024 * 60];
    snprintf(buf, sizeof(buf), u8"%s", strTemp.c_str());
    TCHAR wscBuffer[1024 * 10] = { 0 };
    MultiByteToWideChar(CP_UTF8, 0, buf, (int)strlen(buf) + 1, wscBuffer, sizeof(wscBuffer) / sizeof(wchar_t));
    memset(buf, 0, 1024 * 9);
    WideCharToMultiByte(CP_ACP, 0, wscBuffer, -1, buf, 1024 * 9, NULL, NULL);
    return buf;
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
    float r = static_cast<float>(rand() % 13317) / 13317.0f;
    float* pt = weight;
    return std::lower_bound(pt, pt + 13317, r) - pt;
}

std::map<std::string, int> tokenizer_token2idx;
std::map<int, std::string> tokenizer_idx2token;
std::vector<int> token2idx(std::string token)
{
    std::vector<int> idx;
    std::vector<std::string> spliteList = SpiteStringCharacter(token);
    for (auto s : spliteList)
        idx.push_back(tokenizer_token2idx[s]);
    return idx;
}
std::string idx2token(std::vector<int> idx)
{
    std::string token;
    for (auto i : idx)
        token += tokenizer_idx2token[i];
    return token;
}

int main()
{
    std::srand(static_cast <unsigned> (time(0)));

    std::ifstream infile;
    std::string pathname = "assert/vocab.txt";
    infile.open(pathname.data());
    std::string s;
    int idx = 0;
    while (getline(infile, s)) {
        s = UTF8string(s);
        tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
        tokenizer_idx2token.insert(std::pair<int, std::string>(idx, s));
        idx++;
    }
    infile.close();

    ncnn::Net net;
    net.opt.use_packing_layout = false;
    net.register_custom_layer("DivTrilWhere", DivTrilWhere_layer_creator);
    net.register_custom_layer("Gather", Gather_layer_creator);
    net.load_param("assert/gpt2.param");
    net.load_model("assert/gpt2.bin");

    std::cout << "尽量用中文，目前英文有点小问题，输入quit退出，输入refresh清空记忆" << std::endl;
    
    // 唯二的可配置参数，会影响计算速度
    int max_history_len = 3;
    int max_len = 25;

    std::vector<std::vector<int>> history;
    while (1) {
        std::string text;
        std::cout << "user:";
        std::cin >> text;
        if (text == "quit") break;
        if (text == "refresh") {
            history.clear();
            continue;
        }

        std::vector<int> text_ids = token2idx(text);
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
        std::cout << "chatbot:" << bot_text << std::endl;
    }

    return 0;
}
