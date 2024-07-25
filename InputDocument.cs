using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Rustdocfx;

internal enum InputDocumentKind
{
    Attribute,
    Constant,
    Derive,
    Enum,
    Fn,
    Macro,
    Struct,
    Trait,
    Type,

    Module,
    Crate,
}

internal sealed class InputDocument
{
    public required string AbsolutePath { get; init; }
    public required string RelativePath { get; init; }

    public required InputDocumentKind Kind { get; init; }

    /// <summary>
    /// If true, we need to generate a new output document for this input document.
    /// 
    /// This may be because the input document has changed or is entirely new.
    /// </summary>
    public required bool GenerateOutput { get; init; }
}
