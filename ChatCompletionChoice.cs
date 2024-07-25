using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rustdocfx;

internal sealed class ChatCompletionChoice
{
    public string? FinishReason { get; set; }
    public ChatCompletionMessage? Message { get; set; }
}
