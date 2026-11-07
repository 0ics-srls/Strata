// CPU-only tests for the conversation disk tier's rules (include/strata/core/conversation_disk.hpp): when the tier can
// open, when a read back is worth it, the default cap.  Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no GPU, no model.
#include "strata/core/conversation_disk.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

bool says(const std::string& why, const char* option) { return !why.empty() && why.find(option) != std::string::npos; }

ConversationDiskSetup production() {
    ConversationDiskSetup s;   // ubuntu-ai: --serve, --prompt-cache 6 (the default), the V100 as a peer
    s.serve = true;
    s.prompt_cache = 6;
    s.layer_split = false;
    s.batch = false;
    s.peer_device = 1;
    return s;
}
}  // namespace

int main() {
    check(conversation_disk_refusal(production()).empty(), "production (serve, prompt cache, the V100 as a peer) opens the tier");
    {
        ConversationDiskSetup s = production();
        s.serve = false;
        check(says(conversation_disk_refusal(s), "--serve"), "without --serve: refused, and the reason names it");
    }
    {
        ConversationDiskSetup s = production();
        s.prompt_cache = 0;
        check(says(conversation_disk_refusal(s), "--prompt-cache"), "with --prompt-cache 0: refused, and the reason names it");
    }
    {
        ConversationDiskSetup s = production();
        s.layer_split = true;
        check(says(conversation_disk_refusal(s), "--layer-split"), "with a layer split: refused, and the reason names it");
    }
    {
        ConversationDiskSetup s = production();
        s.batch = true;
        check(says(conversation_disk_refusal(s), "--batch"), "with --batch: refused, and the reason names it");
    }
    {
        ConversationDiskSetup s = production();
        s.peer_device = 0;
        check(conversation_disk_refusal(s).empty(), "one GPU without a peer opens the tier");
    }
    check(conversation_disk_worth(12048, 10000), "2,048 tokens more than the best elsewhere: worth a read back");
    check(!conversation_disk_worth(12047, 10000), "2,047 more: not worth it");
    check(!conversation_disk_worth(9000, 10000), "fewer than the best elsewhere: not worth it");
    check(conversation_disk_worth(2048, 0), "nothing elsewhere: an entry of 2,048 tokens is worth it");
    check(!conversation_disk_worth(2047, 0), "nothing elsewhere: 2,047 is not");
    check(kConversationDiskDefaultGib == 100, "the default cap is 100 GB");
    std::printf("conversation_disk_rules_test: %d checks passed\n", checks);
    return 0;
}
