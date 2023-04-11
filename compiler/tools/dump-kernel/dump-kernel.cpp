#include <cstdint>
#include <numeric>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "assert.h"
#include "mlir/ExecutionEngine/OptUtils.h"

using namespace mlir;
using namespace llvm;
cl::opt<std::string> InputFile(
        cl::Positional, cl::Required, cl::desc("<input mgb model>"));

cl::opt<std::string> FindSym(cl::Positional, cl::Required, cl::desc("<input sym>"));

cl::opt<std::string> OutputFile(
        cl::NormalFormatting, cl::Optional, cl::desc("<output file path, default>"));

static OwningOpRef<ModuleOp> parseMLIRInput(
        StringRef inputFilename, MLIRContext* context) {
    // Set up the input file.
    std::string errorMessage;
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
        llvm::errs() << errorMessage << "\n";
        return nullptr;
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
    return OwningOpRef<ModuleOp>(parseSourceFile(sourceMgr, context));
}

int main(int argc, char** argv) {
    cl::ParseCommandLineOptions(argc, argv);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::DialectRegistry registry;
    registerLLVMDialectTranslation(registry);
    MLIRContext context(registry);
    auto module = parseMLIRInput(InputFile.getValue(), &context);

    auto opt_pipeline = mlir::makeOptimizingTransformer(3, 3, 0);
    std::vector<std::string> libs;
    auto&& mb_engine = mlir::ExecutionEngine::create(
            *module, nullptr, opt_pipeline, llvm::None,
            std::vector<llvm::StringRef>(libs.begin(), libs.end()), true, false);

    assert(mb_engine && "Error can't create engine\n");
    std::unique_ptr<mlir::ExecutionEngine> my_engine = std::move(*mb_engine);

    if (FindSym.getValue().size() > 0) {
        auto lkup_rst = my_engine->lookup(FindSym.getValue());
        assert(lkup_rst && "lkup_rst ");
    }

    std::string dump_obj_file =
            OutputFile.getValue().size() > 0 ? OutputFile.getValue() : "./dump_mlir.o";
    my_engine->dumpToObjectFile(dump_obj_file);
    return 0;
}
