#pragma once

#include "eden.h"

#include <map>
#include <regex>
#include <string>
#include <vector>

struct eden_vocab;

// grammar element type
enum eden_gretype {
    // end of rule definition
    EDEN_GRETYPE_END            = 0,

    // start of alternate definition for rule
    EDEN_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    EDEN_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    EDEN_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    EDEN_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding EDEN_GRETYPE_CHAR or EDEN_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    EDEN_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding EDEN_GRETYPE_CHAR or
    // EDEN_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    EDEN_GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    EDEN_GRETYPE_CHAR_ANY       = 7,

    // terminal element: token (<[token-id]>)
    EDEN_GRETYPE_TOKEN          = 8,

    // inverse token (!<[token-id]>)
    EDEN_GRETYPE_TOKEN_NOT      = 9,
};

typedef struct eden_grammar_element {
    enum eden_gretype type;
    uint32_t           value; // Unicode code point, rule ID, or token ID
} eden_grammar_element;

struct eden_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct eden_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    eden_partial_utf8   partial_utf8;
    eden_token          id;
};

using eden_grammar_rule  = std::vector<      eden_grammar_element>;
using eden_grammar_stack = std::vector<const eden_grammar_element *>;

using eden_grammar_rules      = std::vector<eden_grammar_rule>;
using eden_grammar_stacks     = std::vector<eden_grammar_stack>;
using eden_grammar_candidates = std::vector<eden_grammar_candidate>;

// TODO: remove, needed for tests atm
const eden_grammar_rules  & eden_grammar_get_rules (const struct eden_grammar * grammar);
      eden_grammar_stacks & eden_grammar_get_stacks(      struct eden_grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `eden_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void eden_grammar_accept(struct eden_grammar * grammar, uint32_t chr);

std::vector<eden_grammar_candidate> eden_grammar_reject_candidates_for_stack(
        const eden_grammar_rules      & rules,
        const eden_grammar_stack      & stack,
        const eden_grammar_candidates & candidates);

struct eden_grammar_parser {
    const eden_vocab * vocab;
    std::map<std::string, uint32_t> symbol_ids;

    eden_grammar_rules rules;

    eden_grammar_parser(const struct eden_vocab * vocab = nullptr) : vocab(vocab) {}

    eden_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const eden_grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            eden_grammar_rule & rule,
            bool               is_nested);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct eden_grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;

    size_t find(const std::string & input) const;
};

struct eden_grammar {
    // maintain a list of eden_tokens and their positions in the trigger_buffer
    using token_pos = std::pair<eden_token, std::pair<size_t, size_t>>;

    // note: allow null vocab for testing (not great)
    const eden_vocab * vocab;

    const eden_grammar_rules  rules;  // TODO: shared ptr
          eden_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    eden_partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<token_pos>   trigger_buffer_positions; // Tokens buffered by lazy grammar. Used to replay when a trigger is found.
    std::vector<eden_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<eden_grammar_trigger_pattern>
                             trigger_patterns;         // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                                       // string, and the grammar will be given the string from the first match group onwards.

};

//
// internal API
//

// note: needed for tests (not great)
struct eden_grammar * eden_grammar_init_impl(
        const struct eden_vocab * vocab,
        const eden_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct eden_grammar * eden_grammar_init_impl(
        const struct eden_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const eden_token * trigger_tokens,
                            size_t num_trigger_tokens);

void eden_grammar_free_impl(struct eden_grammar * grammar);

struct eden_grammar * eden_grammar_clone_impl(const struct eden_grammar & grammar);

// TODO: move the API below as member functions of eden_grammar
void eden_grammar_apply_impl(
        const struct eden_grammar & grammar,
            eden_token_data_array * cur_p);

void eden_grammar_accept_impl(
              struct eden_grammar & grammar,
                       eden_token   token);

void eden_grammar_accept_str(
              struct eden_grammar & grammar,
                 const std::string & piece);

void eden_grammar_accept_token(
              struct eden_grammar & grammar,
                       eden_token   token,
                 const std::string & piece);
