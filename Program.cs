using Koek;
using Microsoft.Extensions.Logging;
using Mono.Options;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics;
using System.Reflection.Emit;
using System.Reflection;
using System.Text.Json;
using System.IO.Hashing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices.ObjectiveC;
using System.Text;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using Microsoft.Extensions.DependencyInjection;
using System.Xml.Schema;

namespace Rustdocfx;

public sealed class Program : IDisposable
{
    static async Task Main(string[] args)
    {
        using (var program = new Program())
            await program.ExecuteAsync(args);
    }

    private async Task ExecuteAsync(string[] args)
    {
        if (!ParseArguments(args))
        {
            Environment.ExitCode = -1;
            return;
        }

        Console.CancelKeyPress += OnControlC;

        var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConsole();

            if (_verbose)
                builder.SetMinimumLevel(LogLevel.Debug);
        });
        var logger = loggerFactory.CreateLogger("");

        try
        {
            logger.LogInformation("Identifying input documents.");

            // First, we create a map of all the content we find in the Rustdoc output.
            // This will identify each document, categorize it for processing, identify
            // whether it has even changed and determine the Docfx output filename.
            var inputDocuments = GetInputDocuments(_newHtmlRoot, _oldHtmlRoot, _outputRoot).ToList();

            foreach (var doc in inputDocuments)
            {
                var isQueued = doc.GenerateOutput ? " (generation queued)" : string.Empty;
                logger.LogDebug($"Found {doc.Kind}: {doc.RelativePath}{isQueued}.");
            }

            logger.LogInformation("Removing outdated output documents.");

            // Clean up any output files that do not map to specific input documents.
            // This removes old content for types that no longer exist, as well as any
            // generated content (e.g. tables of contents) that will be re-generated.
            CleanOutput(_outputRoot, inputDocuments);

            logger.LogInformation("Generating new output documents.");

            // Generate output documents for any new or changed input documents.
            await GenerateOutputAsync(_outputRoot, inputDocuments, GetChatHttpClient(), logger);

            // Generate reference documents such as tables of contents.
            var toc = await GenerateTableOfContentsAsync(inputDocuments, _outputRoot, logger);
            await File.WriteAllTextAsync(Path.Combine(_outputRoot, "toc.yml"), toc);

            // All done!
        }
        catch (OperationCanceledException) when (_cancel.IsCancellationRequested)
        {
        }

        var totalCost = _totalInputTokens * InputTokenCostEuros + _totalOutputTokens * OutputTokenCostEuros;

        logger.LogInformation($"All done. Total cost: {totalCost:F2} €");
    }

    // From https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
    private const double InputTokenCostEuros = 0.0000047;
    private const double OutputTokenCostEuros = 0.0000141;

    private HttpClient GetChatHttpClient()
    {
        var services = new ServiceCollection();

        var httpClientBuilder = services.AddHttpClient(
            name: "default",
            configureClient: client =>
            {
                client.BaseAddress = new($"https://{_aiEndpoint}/openai/deployments/{_aiDeployment}/chat/completions?api-version=2024-02-15-preview");
                client.DefaultRequestHeaders.Add("api-key", _aiKey);
                // This conflicts with the resilience strategy if set and causes nonretriable failures too eagerly.
                // Just let the resilience strategy handle all timeout logic.
                client.Timeout = Timeout.InfiniteTimeSpan;
            });

        httpClientBuilder.AddStandardResilienceHandler(options =>
        {
            options.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(20);
            options.AttemptTimeout.Timeout = TimeSpan.FromMinutes(5);
            options.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(10);
        });

        return services.BuildServiceProvider().GetRequiredService<IHttpClientFactory>().CreateClient("default");
    }

    #region Input identification
    private IEnumerable<InputDocument> GetInputDocuments(string newHtmlRoot, string oldHtmlRoot, string outputRoot)
    {
        return GetInputDocumentsInner(newHtmlRoot, newHtmlRoot, oldHtmlRoot, outputRoot);
    }

    /// <summary>
    /// Gets the input documents from a single directory (recursively).
    /// </summary>
    private IEnumerable<InputDocument> GetInputDocumentsInner(string directoryPath, string newHtmlRoot, string oldHtmlRoot, string outputRoot)
    {
        // We special-case the Rustdoc root directory because it contains the list of crates
        // that we will be processing. If we detect we are in the root, we directly process
        // the crate directories and ignore everything else. Note that we do not require the
        // entire process to be necessarily started in the Rustdoc root (maybe the user only
        // wants to convert one specific module, which we can support with minor hassle).
        if (File.Exists(Path.Combine(directoryPath, "crates.js")))
        {
            foreach (var doc in GetInputDocumentFromCrates(directoryPath, newHtmlRoot, oldHtmlRoot, outputRoot))
                yield return doc;

            yield break;
        }

        // We only care about HTML files - the rest may be used here and
        // there on demand but only the HTML contains our valuable content.
        foreach (var file in Directory.GetFiles(directoryPath, "*.html"))
        {
            if (Path.GetFileName(file) == "all.html")
            {
                // Do not care - this is just a list of everything under this node.
                // We will generate our own table of contents, so this is not useful.
                continue;
            }

            var kind = IdentifyInputDocumentKind(file);

            // Some documents are simply redirects to a "canonical" path for the item.
            // We ignore these and only process the canonical file.
            var contents = File.ReadAllText(file);
            if (contents.Contains("<meta http-equiv=\"refresh\" ", StringComparison.Ordinal))
                continue;

            var relativePath = Path.GetRelativePath(newHtmlRoot, file);

            yield return new InputDocument
            {
                AbsolutePath = file,
                RelativePath = relativePath,
                Kind = kind,
                GenerateOutput = ShouldGenerateOutput(relativePath, newHtmlRoot, oldHtmlRoot, outputRoot)
            };
        }

        // Any directories here are submodules (only the Rustdoc root has other directories).
        foreach (var module in Directory.GetDirectories(directoryPath))
        {
            foreach (var doc in GetInputDocumentsInner(module, newHtmlRoot, oldHtmlRoot, outputRoot))
                yield return doc;
        }
    }

    private InputDocumentKind IdentifyInputDocumentKind(string path)
    {
        var filename = Path.GetFileName(path);

        if (filename == "index.html")
        {
            // This is the crate/module index. We need to determine which by looking at the contents.
            // Crate files will contain "<h1>Crate ", whereas module files will contain "<h1>Module ".
            // NOTE: Modules might not have an index file (if author did not write any module docs).
            var content = File.ReadAllText(path);

            if (content.Contains("<h1>Crate ", StringComparison.Ordinal))
                return InputDocumentKind.Crate;
            else if (content.Contains("<h1>Module ", StringComparison.Ordinal))
                return InputDocumentKind.Module;
            else
                throw new InvalidOperationException($"Failed to identify the kind of document at '{path}'.");
        }

        if (filename.StartsWith("attr.", StringComparison.Ordinal))
            return InputDocumentKind.Attribute;
        else if (filename.StartsWith("constant.", StringComparison.Ordinal))
            return InputDocumentKind.Constant;
        else if (filename.StartsWith("derive.", StringComparison.Ordinal))
            return InputDocumentKind.Derive;
        else if (filename.StartsWith("enum.", StringComparison.Ordinal))
            return InputDocumentKind.Enum;
        else if (filename.StartsWith("fn.", StringComparison.Ordinal))
            return InputDocumentKind.Fn;
        else if (filename.StartsWith("macro.", StringComparison.Ordinal))
            return InputDocumentKind.Macro;
        else if (filename.StartsWith("struct.", StringComparison.Ordinal))
            return InputDocumentKind.Struct;
        else if (filename.StartsWith("trait.", StringComparison.Ordinal))
            return InputDocumentKind.Trait;
        else if (filename.StartsWith("type.", StringComparison.Ordinal))
            return InputDocumentKind.Type;

        throw new InvalidOperationException($"Failed to identify the kind of document at '{path}'.");
    }

    private IEnumerable<InputDocument> GetInputDocumentFromCrates(string directoryPath, string newHtmlRoot, string oldHtmlRoot, string outputRoot)
    {
        var cratesJsPath = Path.Combine(directoryPath, "crates.js");
        var cratesJsContent = File.ReadAllText(cratesJsPath);

        // The content is: window.ALL_CRATES = ["geneva_provider","oxidizer","oxidizer_macros","oxidizer_macros_impl","substrate_environment","substrate_server","substrate_win32"];
        // We want to strip the junk at the start and parse the array.
        var start = cratesJsContent.IndexOf('[', StringComparison.Ordinal);
        var end = cratesJsContent.LastIndexOf("]", StringComparison.Ordinal);

        if (start == -1 || end == -1)
            throw new InvalidOperationException("Failed to obtain a list of crates from 'crates.js'.");

        var cratesArrayJson = cratesJsContent[start..(end + 1)];

        var crates = JsonSerializer.Deserialize<string[]>(cratesArrayJson);

        if (crates == null || crates.Length == 0)
            throw new InvalidOperationException("Failed to obtain a list of crates from 'crates.js'.");

        foreach (var crate in crates)
        {
            var crateDirectory = Path.Combine(directoryPath, crate);

            if (!Directory.Exists(crateDirectory))
                throw new InvalidOperationException($"The crate directory '{crateDirectory}' does not exist.");

            foreach (var doc in GetInputDocumentsInner(crateDirectory, newHtmlRoot, oldHtmlRoot, outputRoot))
                yield return doc;
        }
    }

    private bool ShouldGenerateOutput(string relativePath, string newHtmlPath, string oldHtmlPath, string outputRoot)
    {
        var newDocumentPath = Path.Combine(newHtmlPath, relativePath);
        var oldDocumentPath = Path.Combine(oldHtmlPath, relativePath);

        if (!File.Exists(oldDocumentPath))
            return true;

        var oldFileInfo = new FileInfo(oldDocumentPath);
        var newFileInfo = new FileInfo(newDocumentPath);

        if (oldFileInfo.Length != newFileInfo.Length)
            return true;

        // We cannot rely on dates/times because we might just be checking out stuff from a repo
        // in which case the timestamps may be meaningless or misaligned or otherwise untrustworthy.
        // The only option left to us is to compare the files for differences. We will use a hash
        // as a quick and dirty comparison (though we could be more efficient here by comparing directly).
        var oldHash = HashFile(oldDocumentPath);
        var newHash = HashFile(newDocumentPath);

        if (oldHash != newHash)
            return true;

        // If both old and new match, we still generate if the output document is missing.
        return !File.Exists(GetOutputDocumentPath(relativePath, outputRoot));
    }

    private ulong HashFile(string path)
    {
        var hash = new XxHash3();
        using (var stream = File.OpenRead(path))
        {
            hash.Append(stream);
        }

        return hash.GetCurrentHashAsUInt64();
    }
    #endregion

    #region Output generation
    private void CleanOutput(string outputRoot, IEnumerable<InputDocument> inputDocuments)
    {
        // We expect each input file to become one output file with the .md extension.
        // If we are going to generate a file, we will skip it as an allowed file, to delete it ASAP.
        // We will also generate additional files that are re-generated every time (e.g. table of contents).
        var allowedOutputFiles = inputDocuments
            .Where(doc => !doc.GenerateOutput)
            .Select(doc => GetOutputDocumentPath(doc, outputRoot))
            .ToHashSet();

        foreach (var file in Directory.GetFiles(outputRoot, "*.*", SearchOption.AllDirectories))
        {
            if (!allowedOutputFiles.Contains(file))
                File.Delete(file);
        }
    }

    private async Task GenerateOutputAsync(string outputRoot, IEnumerable<InputDocument> inputDocuments, HttpClient chatClient, ILogger logger)
    {
        // We will generate the output files in the same directory structure as the input files.
        // This means we can just replace the root path and the file extension to get the output path.
        foreach (var doc in inputDocuments)
        {
            if (!doc.GenerateOutput)
                continue;

            var outputFilePath = GetOutputDocumentPath(doc, outputRoot);
            Directory.CreateDirectory(Path.GetDirectoryName(outputFilePath)!);
            var outputContent = await GenerateOutputContentAsync(doc, chatClient, logger);
            File.WriteAllText(outputFilePath, outputContent);
        }
    }

    private string GetOutputDocumentPath(InputDocument document, string outputRoot)
    {
        return GetOutputDocumentPath(document.RelativePath, outputRoot);
    }

    private string GetOutputDocumentPath(string relativePath, string outputRoot)
    {
        return Path.ChangeExtension(Path.Combine(outputRoot, relativePath), "md");
    }

    private static readonly JsonSerializerOptions JsonSerializerOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    private long _totalInputTokens;
    private long _totalOutputTokens;

    private static readonly string SystemPrompt = File.ReadAllText("SystemPrompt.txt");

    private const string TagHtmlInputStart = "== HTML INPUT START ==";
    private const string TagHtmlInputEnd = "== HTML INPUT END ==";
    private const string TagMarkdownInputStart = "== MARKDOWN INPUT START ==";
    private const string TagMarkdownInputEnd = "== MARKDOWN INPUT END ==";
    private const string TagAppend = "== APPEND ==";
    private const string TagFinal = "== FINAL ==";
    private const string TagError = "== ERROR ==";

    /// <summary>
    /// Safety stop - if we get stuck in some infinite loop, we abort with an error instead of burning a pile of money.
    /// </summary>
    private const int MaxRounds = 25;

    private async Task<string> GenerateOutputContentAsync(InputDocument document, HttpClient chatClient, ILogger logger)
    {
        logger.LogInformation($"Generating Markdown version of {document.RelativePath}");

        var round = 0;

        // We collect the output here, piece by piece, as we get it. We need to collect it because we might not
        // get the entire output in one response due to response size limits.
        var outputBuilder = new StringBuilder();

        while (true)
        {
            var inputBuilder = new StringBuilder();
            inputBuilder.AppendLine(TagHtmlInputStart);
            inputBuilder.AppendLine(File.ReadAllText(document.AbsolutePath));
            inputBuilder.AppendLine(TagHtmlInputEnd);
            inputBuilder.AppendLine(TagMarkdownInputStart);
            inputBuilder.Append(outputBuilder);
            inputBuilder.AppendLine(TagMarkdownInputEnd);

            var payload = new
            {
                messages = new object[]
                {
                    new
                    {
                        role = "system",
                        content = new object[]
                        {
                            new
                            {
                                type = "text",
                                text = SystemPrompt
                            }
                        }
                    },
                    new
                    {
                        role = "user",
                        content = new object[]
                        {
                            new
                            {
                                type = "text",
                                text = inputBuilder.ToString()
                            }
                        }
                    }
                },
                // Values from sample code, unknown how much sense they make.
                temperature = 0.7,
                top_p = 0.95,
                // This depends on model. We assume ChatGPT 4o, which has a 4096 output token limit.
                max_tokens = 4096,
                // Just give the response as one blob, no need to feed it to us piece by piece.
                stream = false,
            };

            var serializedPayload = JsonSerializer.SerializeToUtf8Bytes(payload, JsonSerializerOptions);
            logger.LogDebug(Encoding.UTF8.GetString(serializedPayload));

            var requestContent = new ByteArrayContent(serializedPayload);
            requestContent.Headers.ContentType = new MediaTypeHeaderValue("application/json");

            var response = await chatClient.PostAsync("", requestContent, _cancel);
            response.EnsureSuccessStatusCode();

            var responseString = await response.Content.ReadAsStringAsync(_cancel);

            logger.LogDebug(responseString);

            var responsePayload = JsonSerializer.Deserialize<ChatCompletionResponse>(responseString, JsonSerializerOptions);

            _totalInputTokens += responsePayload?.Usage?.PromptTokens ?? 0;
            _totalOutputTokens += responsePayload?.Usage?.CompletionTokens ?? 0;

            logger.LogDebug($"Cumulative input tokens: {_totalInputTokens}, output tokens: {_totalOutputTokens}");

            var responseMessage = responsePayload?.Choices?.FirstOrDefault()?.Message?.Content;

            if (responseMessage == null)
                throw new ContractException("Failed to obtain a response from the AI model - no message could be parsed from response.");

            // We expect the response to start with the "Append" tag. It may or may not contain the "Final" tag.
            // The message may also, at any place, contain the "Error" tag. We look for this first of all and
            // do nothing else if we encounter an error.
            if (responseMessage.Contains(TagError, StringComparison.Ordinal))
                throw new ContractException("The AI model reported an error: " + responseMessage);

            // The response may contain both types of newlines. We split by Windows newlines first to get rid
            // of them, and by Linux newlines later to catch any stragglers. This way we cover both types of newlines.
            var responseMessageLines = responseMessage
                .Split("\r\n")
                .SelectMany(x => x.Split('\n'))
                .ToList();

            bool isFinal = responseMessageLines.LastOrDefault() == TagFinal;

            if (isFinal)
                responseMessageLines.RemoveAt(responseMessageLines.Count - 1);

            if (responseMessageLines.FirstOrDefault() == TagAppend)
            {
                responseMessageLines.RemoveAt(0);

                // Whatever is left in the buffer is the content to be appended.
                foreach (var line in responseMessageLines)
                    outputBuilder.AppendLine(line);
            }
            else
            {
                // Maybe the AI was shy and did not emit "final" when it should have? Legit scenario.
                if (responseMessageLines.Count != 0)
                    throw new ContractException("The AI model returned unexpected content in the response message.");
            }

            if (isFinal)
                return outputBuilder.ToString();

            round++;

            if (round > MaxRounds)
                throw new ContractException("The AI model did not return a final response after the maximum number of rounds.");

            logger.LogDebug($"Incomplete output for {document.RelativePath} - proceeding with next round.");
        }

        throw new UnreachableCodeException();
    }
    #endregion

    #region Auxilliary file generation
    async Task<string> GenerateTableOfContentsAsync(List<InputDocument> inputDocuments, string outputRoot, ILogger logger)
    {
        // Desired format:
        //
        // Crate X
        //  Module foo
        //   Struct A
        //   Struct B
        //   Type C
        //  Module bar
        //
        // Each container-like node is represented by index.html. We assume that all crate/module directories
        // contain an index.html, as this seems to always be present for crates/modules in Rustdoc output.

        // We will not make assumptions about the order of input documents, so we first need to establish relations
        // between them. We establish two data sets:
        // * If a document qualifies as a container-like document (because it is named index.html),
        //   establish the relative path that children can use to lookup this parent document (i.e. the directory name).
        // * For each parent document, establish a list of children that can be attached to it.
        var relativePathToParent = new Dictionary<string, InputDocument>();
        var parentToChildren = new Dictionary<InputDocument, List<InputDocument>>();

        foreach (var doc in inputDocuments)
        {
            if (doc.Kind != InputDocumentKind.Crate && doc.Kind != InputDocumentKind.Module)
            {
                // This document cannot be a container.
                continue;
            }

            relativePathToParent[Path.GetDirectoryName(doc.RelativePath) ?? ""] = doc;
            parentToChildren[doc] = new List<InputDocument>();
        }

        // Now we can simply go through every document and attach it to its rightful parent. That creates our document tree.
        var topLevelDocuments = new List<InputDocument>();

        foreach (var doc in inputDocuments)
        {
            var parent = Path.GetDirectoryName(doc.RelativePath) ?? "";

            // If the item itself is already a parent, we need to go up 1 level.
            if (parentToChildren.ContainsKey(doc))
            {
                parent = Path.GetDirectoryName(parent);

                if (parent == null)
                {
                    logger.LogDebug($"{doc.RelativePath} is a top level document with no parent.");
                    topLevelDocuments.Add(doc);
                    continue;
                }
            }

            if (!relativePathToParent.TryGetValue(parent, out var parentDocument))
            {
                logger.LogDebug($"{doc.RelativePath} is promoted to top level document because it has no parent.");
                topLevelDocuments.Add(doc);
                continue;
            }

            logger.LogDebug($"Parent ID of {doc.RelativePath} is '{parent}'");
            parentToChildren[parentDocument].Add(doc);
        }

        // Now we have our tree of documents and all we need to do is transform it to a tree of TOC nodes.
        var topLevelTocNodes = await ToTocNodesAsync(topLevelDocuments, parentToChildren, outputRoot);

        // Conert this to DocFX TOC format.
        var toc = new StringBuilder();
        AppendTocNodes(toc, level: 0, topLevelTocNodes, outputRoot);

        return toc.ToString();
    }

    private void AppendTocNodes(StringBuilder toc, int level, List<TocNode> nodes, string outputRoot)
    {
        // Format:
        //- name: Service
        //  href: Service/
        //- name: Publishing
        //  href: Publishing/
        //- name: Operations
        //  href: Operations/
        //- name: Player
        //  href: Player/
        //- name: Resource library
        //  href: Library/
        //- name: Introduction
        //  href: Introduction.md
        //- name: Glossary
        //  href: Glossary.md
        //- name: Records and logs
        //  href: Records/toc.yml
        //- name: Technical articles
        //  href: TechnicalArticles/toc.yml
        //- name: Threat model
        //  href: ThreatModel/ThreatModel.md

        foreach (var node in nodes.OrderBy(x => x.Name, StringComparer.InvariantCultureIgnoreCase))
        {
            var indentation = new string(' ', level * 2);
            toc.Append(indentation);
            toc.AppendLine($"- name: {node.Name}");

            var outputRelativeUrl = Path.GetRelativePath(outputRoot, GetOutputDocumentPath(node.Document, outputRoot)).Replace('\\', '/');
            toc.Append(indentation);
            toc.AppendLine($"  href: {outputRelativeUrl}");

            // If we have child items, we emit those as an "items" array under this TOC node.
            if (node.Children.Count != 0)
            {
                toc.Append(indentation);
                toc.AppendLine("  items:");

                AppendTocNodes(toc, level + 1, node.Children, outputRoot);
            }
        }
    }

    private async Task<List<TocNode>> ToTocNodesAsync(IEnumerable<InputDocument> documents, Dictionary<InputDocument, List<InputDocument>> childrenMap, string outputRoot)
    {
        var result = new List<TocNode>();

        foreach (var doc in documents)
            result.Add(await ToTocNodeAsync(doc, childrenMap, outputRoot));

        return result;
    }

    private async Task<TocNode> ToTocNodeAsync(InputDocument doc, Dictionary<InputDocument, List<InputDocument>> childrenMap, string outputRoot)
    {
        var node = new TocNode
        {
            Name = await GetTocNodeNameAsync(doc, outputRoot),
            Document = doc,
        };

        if (childrenMap.TryGetValue(doc, out var children))
            node.Children = await ToTocNodesAsync(children, childrenMap, outputRoot);

        return node;
    }

    private sealed class TocNode
    {
        public required string Name { get; init; }
        public required InputDocument Document { get; init; }
        public List<TocNode> Children { get; set; } = new();
    }

    private static readonly string[] TocNameParseHeadingPartSeparators = new[] { "::", " " };

    private async Task<string> GetTocNodeNameAsync(InputDocument doc, string outputRoot)
    {
        // We get the name from the output document because it is easier to process Markdown without all the HTML fluff.
        // The first heading is going to be something like:
        // # Derive Macro oxidizer_macros::Context
        //
        // We want the "Context" from here, so we just find the last "::" or " " and take whatever follows.
        var outputContent = await File.ReadAllLinesAsync(GetOutputDocumentPath(doc, outputRoot));

        // There may be empty lines before the first heading. Who knows what else there may be - just find the first heading.
        var headingLine = outputContent.FirstOrDefault(x => x.StartsWith("# ", StringComparison.Ordinal));
        var lastPart = headingLine?.Split(TocNameParseHeadingPartSeparators, StringSplitOptions.RemoveEmptyEntries).LastOrDefault();

        // If we failed, just use the filename and let the humans figure out what is wrong when they see it.
        if (lastPart == null)
            return Path.GetFileNameWithoutExtension(doc.RelativePath);

        // We prefix the last part of the heading with the type of the node.
        return $"{doc.Kind} {lastPart}";
    }
    #endregion

    private void OnControlC(object? sender, ConsoleCancelEventArgs e)
    {
        _cts.Cancel();
        e.Cancel = true; // We have handled it.
    }

    private Program()
    {
        _cancel = _cts.Token;
    }

    private readonly CancellationTokenSource _cts = new();
    private readonly CancellationToken _cancel;

    public void Dispose()
    {
        _cts.Cancel();
        _cts.Dispose();
    }

    private string? _oldHtmlRoot;
    private string? _newHtmlRoot;
    private string? _outputRoot;
    private string? _aiKey;
    private string? _aiEndpoint;
    private string? _aiDeployment;
    private bool _verbose;

    [MemberNotNullWhen(true, nameof(_oldHtmlRoot))]
    [MemberNotNullWhen(true, nameof(_newHtmlRoot))]
    [MemberNotNullWhen(true, nameof(_outputRoot))]
    [MemberNotNullWhen(true, nameof(_aiKey))]
    [MemberNotNullWhen(true, nameof(_aiDeployment))]
    private bool ParseArguments(string[] args)
    {
        var showHelp = false;
        var debugger = false;

        var options = new OptionSet
            {
                $"Usage: {Assembly.GetExecutingAssembly().GetName().Name}.exe --old-html docs-previous/ --new-html docs-current/ --out website/api-docs/ --ai-key 000000000000000000000000 --ai-endpoint my-azure-openai-endpoint.example.com --ai-deployment my-ai-deployment",
                "Converts Rustdoc HTML output to equivalent Docfx markdown pages.",
                "",
                { "h|?|help", "Displays usage instructions.", val => showHelp = val != null },
                "",
                "General",
                { "old-html=", "Path to the previous version of the Rustdoc HTML output. Must exist but may be an empty directory if there is no previous version.", (string val) => _oldHtmlRoot = val },
                { "new-html=", "Path to the current version of the Rustdoc HTML output. Must exist.", (string val) => _newHtmlRoot = val },
                { "out=", "Path to the directory to update with new Docfx output files. It is assumed that this directory contains the previous output from running this tool against old-html.", (string val) => _outputRoot = val },
                "OpenAI",
                { "ai-key=", "API key to access the Azure OpenAI resource.", (string val) => _aiKey = val },
                { "ai-endpoint=", "DNS name of the Azure OpenAI resource, without any https:// prefix or URL path.", (string val) => _aiEndpoint = val },
                { "ai-deployment=", "Name the ChatGPT deployment within the Azure OpenAI resource.", (string val) => _aiDeployment = val },
                "Diagnostics",
                { "verbose", "Enables verbose logging.", val => _verbose = val != null },
                { "debugger", "Requests a debugger to be attached before data processing starts.", val => debugger = val != null, true }
            };

        List<string> remainingOptions;

        try
        {
            remainingOptions = options.Parse(args);

            if (args.Length == 0 || showHelp)
            {
                options.WriteOptionDescriptions(Console.Out);
                return false;
            }

            if (string.IsNullOrWhiteSpace(_oldHtmlRoot))
                throw new OptionException("The --old-html parameter is required.", "old-html");

            if (!Directory.Exists(_oldHtmlRoot))
                throw new OptionException("The --old-html parameter must reference an existing directory, even if empty.", "old-html");

            if (string.IsNullOrWhiteSpace(_newHtmlRoot))
                throw new OptionException("The --new-html parameter is required.", "new-html");

            if (!Directory.Exists(_newHtmlRoot))
                throw new OptionException("The --new-html parameter must reference an existing directory.", "new-html");

            if (string.IsNullOrWhiteSpace(_outputRoot))
                throw new OptionException("The --out parameter is required.", "out");

            if (!Directory.Exists(_outputRoot))
                throw new OptionException("The --out parameter must reference an existing directory, even if empty.", "out");

            if (string.IsNullOrWhiteSpace(_aiKey))
                throw new OptionException("The --ai-key parameter is required.", "ai-key");

            if (string.IsNullOrWhiteSpace(_aiEndpoint))
                throw new OptionException("The --ai-endpoint parameter is required.", "ai-endpoint");

            if (string.IsNullOrWhiteSpace(_aiDeployment))
                throw new OptionException("The --ai-deployment parameter is required.", "ai-deployment");

            // Convert all to absolute paths to canonize them for easier processing later on.
            _oldHtmlRoot = Path.GetFullPath(_oldHtmlRoot);
            _newHtmlRoot = Path.GetFullPath(_newHtmlRoot);
            _outputRoot = Path.GetFullPath(_outputRoot);
        }
        catch (OptionException ex)
        {
            Console.WriteLine(ex.Message);
            Console.WriteLine("For usage instructions, use the --help command line parameter.");
            return false;
        }

        if (remainingOptions.Count != 0)
        {
            Console.WriteLine("Unknown command line parameters: {0}", string.Join(" ", remainingOptions.ToArray()));
            Console.WriteLine("For usage instructions, use the --help command line parameter.");
            return false;
        }

        if (debugger)
            Debugger.Launch();

        return true;
    }
}
