#include <drogon/WebSocketController.h>
#include <json/json.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <pyreth/server/binary_protocol.hpp>
#include <pyreth/server/protocol.hpp>
#include <pyreth/server/streaming.hpp>
#include <span>
#include <string>

namespace pyreth::server {

std::string formatSSEEvent(std::string_view event, std::string_view data) {
  std::string frame;
  frame.reserve(event.size() + data.size() + 16);
  frame.append("event: ");
  frame.append(event);
  frame.append("\ndata: ");
  frame.append(data);
  frame.append("\n\n");
  return frame;
}

WebSocketServerContext& WebSocketServerContext::instance() noexcept {
  static WebSocketServerContext ctx;
  return ctx;
}

/**
 * @class InferenceWebSocketController
 * @brief Drogon WebSocket controller handling duplex inference at /v1/models/{model_name}/ws.
 */
class InferenceWebSocketController
    : public drogon::WebSocketController<InferenceWebSocketController> {
public:
  void handleNewMessage(const drogon::WebSocketConnectionPtr& wsConnPtr, std::string&& message,
                        const drogon::WebSocketMessageType& type) override {
    if (!wsConnPtr || wsConnPtr->disconnected()) {
      return;
    }

    auto model_name_sp = wsConnPtr->getContext<std::string>();
    std::string model_name = model_name_sp ? *model_name_sp : "";

    auto& ctx = WebSocketServerContext::instance();
    auto* registry = ctx.registry();
    auto* memory_pool = ctx.memoryPool();
    auto* thread_pool = ctx.threadPool();

    if (!registry || !memory_pool || !thread_pool) {
      Json::Value err;
      err["error"] = "Server context not initialized";
      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      wsConnPtr->send(Json::writeString(writer, err));
      return;
    }

    auto entry_res = registry->getModelEntry(model_name);
    if (!entry_res.isOk()) {
      Json::Value err;
      err["error"] = "Model '" + model_name + "' not found";
      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      wsConnPtr->send(Json::writeString(writer, err));
      return;
    }

    auto session = entry_res.value().session;
    if (!session) {
      Json::Value err;
      err["error"] = "Model '" + model_name + "' has no session configured";
      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      wsConnPtr->send(Json::writeString(writer, err));
      return;
    }

    // Binary frame handling
    if (type == drogon::WebSocketMessageType::Binary) {
      auto msg_holder = std::make_shared<std::string>(std::move(message));
      std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(msg_holder->data()),
                                     msg_holder->size());
      auto tensor_res = deserializeBinaryTensor(bytes, msg_holder);
      if (!tensor_res.isOk()) {
        Json::Value err;
        err["error"] = std::string(tensor_res.error().message());
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        wsConnPtr->send(Json::writeString(writer, err));
        return;
      }

      std::vector<Tensor> input_tensors;
      input_tensors.push_back(std::move(tensor_res.value()));

      auto submit_res = thread_pool->submit(
          [wsConnPtr, session, tensors = std::move(input_tensors), msg_holder]() mutable {
            if (!wsConnPtr || wsConnPtr->disconnected()) {
              return;
            }

            auto run_res = session->run(tensors);
            if (!run_res.isOk()) {
              Json::Value err;
              err["error"] = std::string(run_res.error().message());
              Json::StreamWriterBuilder writer;
              writer["indentation"] = "";
              wsConnPtr->send(Json::writeString(writer, err));
              return;
            }

            const auto& out_tensors = run_res.value();
            if (out_tensors.empty()) {
              Json::Value err;
              err["error"] = "Model produced no outputs";
              Json::StreamWriterBuilder writer;
              writer["indentation"] = "";
              wsConnPtr->send(Json::writeString(writer, err));
              return;
            }

            auto ser_bin = serializeBinaryTensor(out_tensors[0]);
            if (!ser_bin.isOk()) {
              Json::Value err;
              err["error"] = std::string(ser_bin.error().message());
              Json::StreamWriterBuilder writer;
              writer["indentation"] = "";
              wsConnPtr->send(Json::writeString(writer, err));
              return;
            }

            const auto& bin_data = ser_bin.value();
            wsConnPtr->send(reinterpret_cast<const char*>(bin_data.data()), bin_data.size(),
                            drogon::WebSocketMessageType::Binary);
          });
      (void)submit_res;
      return;
    }

    // Text / JSON frame handling
    std::string msg_copy = std::move(message);
    auto submit_res = thread_pool->submit(
        [wsConnPtr, session, memory_pool, model_name, body = std::move(msg_copy)]() mutable {
          if (!wsConnPtr || wsConnPtr->disconnected()) {
            return;
          }

          auto lease = memory_pool->acquire();
          ArenaAllocator& arena = lease.get();

          auto req_res = parseKServeRequest(body, arena);
          if (!req_res.isOk()) {
            Json::Value err;
            err["error"] = std::string(req_res.error().message());
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            wsConnPtr->send(Json::writeString(writer, err));
            return;
          }

          auto tensors = req_res.value().extractTensors();

          auto run_res = session->run(tensors);
          if (!run_res.isOk()) {
            Json::Value err;
            err["error"] = std::string(run_res.error().message());
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            wsConnPtr->send(Json::writeString(writer, err));
            return;
          }

          auto out_names = session->getOutputNames();
          auto ser_res = serializeKServeResponse(model_name, out_names, run_res.value(), "1");
          if (!ser_res.isOk()) {
            Json::Value err;
            err["error"] = std::string(ser_res.error().message());
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            wsConnPtr->send(Json::writeString(writer, err));
            return;
          }

          wsConnPtr->send(ser_res.value(), drogon::WebSocketMessageType::Text);
        });
    (void)submit_res;
  }

  void handleNewConnection(const drogon::HttpRequestPtr& req,
                           const drogon::WebSocketConnectionPtr& wsConnPtr) override {
    // Path format: /v1/models/{model_name}/ws
    std::string path = req->path();
    std::string prefix = "/v1/models/";
    std::string suffix = "/ws";

    std::string model_name;
    if (path.starts_with(prefix) && path.ends_with(suffix)) {
      model_name = path.substr(prefix.size(), path.size() - prefix.size() - suffix.size());
    }

    wsConnPtr->setContext(std::make_shared<std::string>(model_name));
  }

  void handleConnectionClosed(const drogon::WebSocketConnectionPtr& /*wsConnPtr*/) override {}

  WS_PATH_LIST_BEGIN
  WS_ADD_PATH_VIA_REGEX(R"(/v1/models/([^/]+)/ws)");
  WS_PATH_LIST_END
};

}  // namespace pyreth::server
