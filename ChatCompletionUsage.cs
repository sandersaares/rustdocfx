using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rustdocfx;

internal sealed class ChatCompletionUsage
{
    public long CompletionTokens { get; set; }
    public long PromptTokens { get; set; }
    public long TotalTokens { get; set; }
}
