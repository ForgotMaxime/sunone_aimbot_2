// Focused config-migration tests (complements tests/node_config_bridge_tests.cpp).
// - defaults() validate cleanly and survive a serialize/deserialize roundtrip.
// - A legacy (v1) workspace migrates to CurrentFormatVersion.
// - Node-owned tuning survives migration instead of being reset to defaults.
#include "nodes/node_graph.h"
#include "nodes/node_config_bridge.h"
#include "config/config.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
using namespace NodeGraph;
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

const Node& findNode(const Workspace& workspace, NodeKind kind)
{
    for (const auto& graph : workspace.graphs)
        for (const auto& node : graph.nodes)
            if (node.kind == kind)
                return node;
    throw std::runtime_error("Expected node missing");
}

Node& findNode(Workspace& workspace, NodeKind kind)
{
    for (auto& graph : workspace.graphs)
        for (auto& node : graph.nodes)
            if (node.kind == kind)
                return node;
    throw std::runtime_error("Expected node missing");
}

void testDefaultsValidateAndRoundtrip()
{
    Workspace workspace = defaults();
    std::string error;
    require(validate(workspace, error), error.c_str());
    findNode(workspace, NodeKind::PredictionTime).parameters["lookahead"] = 0.33;
    std::string text;
    require(serialize(workspace, text, error), error.c_str());
    Workspace loaded;
    require(deserialize(text, loaded, error), error.c_str());
    require(loaded.sourceVersion == CurrentFormatVersion, "Roundtrip lost the format version");
    require(parameter(findNode(loaded, NodeKind::PredictionTime), "lookahead") == 0.33,
        "Roundtrip lost node-owned tuning");
    require(validate(loaded, error), error.c_str());
}

void testLegacyWorkspaceMigrates()
{
    Config config{};
    Workspace workspace;
    workspace.sourceVersion = 1;
    for (std::size_t i = 0; i < GraphCount; ++i)
        workspace.graphs[i].kind = static_cast<GraphKind>(i);
    migrateLegacySettings(workspace, config);
    require(workspace.sourceVersion == CurrentFormatVersion, "Legacy workspace was not migrated");
    std::string error;
    require(validate(workspace, error), error.c_str());
}

void testMigrationPreservesNodeTuning()
{
    Config config{};
    Workspace workspace = defaults();
    findNode(workspace, NodeKind::PredictionTime).parameters["lookahead"] = 0.42;
    migrateLegacySettings(workspace, config);
    require(parameter(findNode(workspace, NodeKind::PredictionTime), "lookahead") == 0.42,
        "Migration reset node-owned tuning");
}
} // namespace

int main()
{
    try
    {
        testDefaultsValidateAndRoundtrip();
        testLegacyWorkspaceMigrates();
        testMigrationPreservesNodeTuning();
        std::cout << "config migrate tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
