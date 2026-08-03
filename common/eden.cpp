#ifdef EDEN_EDEN_INTEGRATION
#include "eden-common.h"
#include <zmq.hpp>
#include <chrono>
#include <iostream>
static zmq::context_t* g_ctx = nullptr;
static zmq::socket_t* g_pub = nullptr;
static bool g_ok = false;
bool eden_init() {
    try { g_ctx = new zmq::context_t(1); g_pub = new zmq::socket_t(*g_ctx, zmq::socket_type::pub);
        g_pub->connect("tcp://127.0.0.1:5556"); g_ok = true; return true; }
    catch (...) { g_ok = false; return false; }
}
static void publish(const char* topic, const std::string& json) {
    if (!g_ok) return;
    try { g_pub->send(zmq::message_t(topic, strlen(topic)), zmq::send_flags::sndmore);
        g_pub->send(zmq::message_t(json), zmq::send_flags::none); } catch (...) {}
}
void eden_publish_inference_start(const std::string& m, uint64_t rid, int32_t pt) {
    publish("inference.started", "{\"model\":\""+m+"\",\"request\":"+std::to_string(rid)+",\"prompt_tokens\":"+std::to_string(pt)+"}");
}
void eden_publish_inference_complete(const std::string& m, uint64_t rid, int32_t pt, int32_t ct, int32_t ms, float tps) {
    publish("inference.completed", "{\"model\":\""+m+"\",\"request\":"+std::to_string(rid)+",\"prompt_tokens\":"+std::to_string(pt)+",\"completion_tokens\":"+std::to_string(ct)+",\"elapsed_ms\":"+std::to_string(ms)+",\"tok_per_sec\":"+std::to_string(tps)+"}");
}
#endif
