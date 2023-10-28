#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"

#include "compiler/Common/Logger.h"
#include "compiler/Common/Version.h"
#include "compiler/Target/onnx/import.h"

using namespace llvm;

cl::opt<std::string> InputFile(
        cl::Positional, cl::Required, cl::desc("<input mgb model>"));
cl::opt<std::string> OutputFile(
        cl::Positional, cl::Required, cl::desc("<output mlir file>"));
cl::opt<bool> Verbose(
        "verbose", cl::desc("log more detail information when compiler model"));

int main(int argc, char** argv) {
    cl::AddExtraVersionPrinter(
            [](raw_ostream& oss) { oss << megcc::getMegccVersionString(); });
    cl::ParseCommandLineOptions(argc, argv);
    if (Verbose) {
        megcc::SetLogLevel(megcc::LogLevel::DEBUG);
    }
    mlir::MLIRContext ctx;
    llvm::SmallVector<llvm::StringRef> names;
    llvm::SplitString(OutputFile, names, ".");
    llvm::outs() << "Import onnx model from " << InputFile.getValue() << "\n";
    mlir::OwningOpRef<mlir::ModuleOp> mod =
            mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::LogicalResult status =
            mlir::ONNX::import_onnx(mod.get(), InputFile.getValue());
    if (mlir::failed(status)) {
        llvm::errs() << "import onnx model failed\n";
        return -1;
    }
    std::error_code EC;
    llvm::raw_fd_stream FileStream(OutputFile.getValue(), EC);
    llvm::outs() << "Export mgb dialect to " << OutputFile.getValue() << "\n";
    mod->print(FileStream);
    llvm::outs() << "onnx convert to mgb dialect done.\n";
    return 0;
}