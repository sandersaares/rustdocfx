using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rustdocfx;

internal sealed class ChatCompletionResponse
{
    public List<ChatCompletionChoice>? Choices { get; set; }
    public ChatCompletionUsage? Usage { get; set; }
}

/*{
  "choices": [
    {
      "content_filter_results": {
        "hate": {
          "filtered": false,
          "severity": "safe"
        },
        "protected_material_code": {
          "filtered": false,
          "detected": false
        },
        "protected_material_text": {
          "filtered": false,
          "detected": false
        },
        "self_harm": {
          "filtered": false,
          "severity": "safe"
        },
        "sexual": {
          "filtered": false,
          "severity": "safe"
        },
        "violence": {
          "filtered": false,
          "severity": "safe"
        }
      },
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "== APPEND ==\n.default_database(\"my_database\".to_string())\r\n.isolation_level(\"read_committed\".to_string())\r\n.build()?;\r\n```\r\n\r\n## Configuration integration\r\n\r\nIf the `config` flag is present...== FINAL ==",
        "role": "assistant"
      }
    }
  ],
  "created": 1721892906,
  "id": "chatcmpl-9onIA8xZMb0N6HHPSQxHrK5nLWrAG",
  "model": "gpt-4o-2024-05-13",
  "object": "chat.completion",
  "prompt_filter_results": [
    {
      "prompt_index": 0,
      "content_filter_results": {
        "hate": {
          "filtered": false,
          "severity": "safe"
        },
        "jailbreak": {
          "filtered": false,
          "detected": false
        },
        "self_harm": {
          "filtered": false,
          "severity": "safe"
        },
        "sexual": {
          "filtered": false,
          "severity": "safe"
        },
        "violence": {
          "filtered": false,
          "severity": "safe"
        }
      }
    }
  ],
  "system_fingerprint": "fp_abc28019ad",
  "usage": {
    "completion_tokens": 275,
    "prompt_tokens": 4373,
    "total_tokens": 4648
  }
}*/