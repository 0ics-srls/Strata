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
    // the resumable length: past it the client compacts the conversation before it could be resumed
    check(conversation_disk_max_tokens(-1, 262144) == 221184, "no option: the context minus 40,960 (221,184 of 262,144)");
    check(conversation_disk_max_tokens(-5, 131072) == 131072 - 40960, "any negative option is the automatic limit");
    check(conversation_disk_max_tokens(30000, 262144) == 30000, "a positive option is the limit");
    check(conversation_disk_max_tokens(0, 262144) == 0, "0: no limit");
    check(conversation_disk_max_tokens(-1, 32768) == 1, "a context under 40,960: the automatic limit is 1, not 0 (no limit)");
    check(conversation_disk_keeps(221184, 221184), "exactly the limit: written");
    check(!conversation_disk_keeps(221185, 221184), "one token past the limit: not written");
    check(!conversation_disk_keeps(2, 1), "a limit of 1 keeps nothing longer than a token");
    check(conversation_disk_keeps(10'000'000, 0), "no limit: always written");
    std::printf("conversation_disk_rules_test: %d checks passed\n", checks);
    return 0;
}
