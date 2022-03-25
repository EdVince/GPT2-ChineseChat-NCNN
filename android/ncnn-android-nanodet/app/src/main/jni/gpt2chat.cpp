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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <locale>
#include <codecvt>
#include <fstream>
#include <map>

#include <platform.h>
#include <benchmark.h>

#include "gpt2.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO , "read txt", __VA_ARGS__)

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static GPT2* g_nanodet = 0;
static ncnn::Mutex lock;

static std::string UTF16StringToUTF8String(const char16_t* chars, size_t len) {
    std::u16string u16_string(chars, len);
    return std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>{}
            .to_bytes(u16_string);
}

std::string JavaStringToString(JNIEnv* env, jstring str) {
    if (env == nullptr || str == nullptr) {
        return "";
    }
    const jchar* chars = env->GetStringChars(str, NULL);
    if (chars == nullptr) {
        return "";
    }
    std::string u8_string = UTF16StringToUTF8String(
            reinterpret_cast<const char16_t*>(chars), env->GetStringLength(str));
    env->ReleaseStringChars(str, chars);
    return u8_string;
}

static std::u16string UTF8StringToUTF16String(const std::string& string) {
    return std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>{}
            .from_bytes(string);
}

jstring StringToJavaString(JNIEnv* env, const std::string& u8_string) {
    std::u16string u16_string = UTF8StringToUTF16String(u8_string);
    auto result =env->NewString(reinterpret_cast<const jchar*>(u16_string.data()),
                                u16_string.length());
    return result;
}

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_edvince_gpt2chatbot_GPT2_loadGPT2(JNIEnv* env, jobject thiz, jobject assetManager, jstring jvocab)
{
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    std::string vocab = JavaStringToString(env,jvocab);

    {
        ncnn::MutexLockGuard g(lock);
        if (!g_nanodet)
            g_nanodet = new GPT2;
        g_nanodet->load(mgr,vocab);
    }



    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_edvince_gpt2chatbot_GPT2_chat(JNIEnv* env, jobject thiz, jstring in)
{
    std::string cpp_in = JavaStringToString(env, in);

    std::string cpp_out = g_nanodet->chat(cpp_in);

    jstring java_out = StringToJavaString(env,cpp_out);

    return java_out;
}

}
