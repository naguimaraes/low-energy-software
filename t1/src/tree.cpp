#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <numeric>
#include <tuple>
#include <cstdlib>
#include <chrono>

// Structure to store verse information
struct VerseInfo {
    std::string book;
    int chapter;
    int verse;
    std::string text;
    
    VerseInfo() : chapter(0), verse(0) {}
    VerseInfo(const std::string& b, int ch, int v, const std::string& t) 
        : book(b), chapter(ch), verse(v), text(t) {}
};

// Structure to store player information
struct PlayerInfo {
    std::string id;
    std::string name;
    std::string position;
    PlayerInfo() = default;
    PlayerInfo(std::string i, std::string n, std::string p)
        : id(std::move(i)), name(std::move(n)), position(std::move(p)) {}
};

// ==================== TRIE IMPLEMENTATION ====================

class TrieNode {
public:
    static const int ALPHABET_SIZE = 26;
    TrieNode* children[ALPHABET_SIZE];
    bool is_end_of_word;
    std::vector<VerseInfo> occurrences;
    
    TrieNode() : is_end_of_word(false) {
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            children[i] = nullptr;
        }
    }
    
    ~TrieNode() {
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            if (children[i] != nullptr) {
                delete children[i];
            }
        }
    }
};

class Trie {
private:
    TrieNode* root;
    
    int char_to_index(char c) const {
        return tolower(c) - 'a';
    }
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    ~Trie() {
        delete root;
    }
    
    void insert(const std::string& word, const VerseInfo& info) {
        TrieNode* current = root;
        
        for (char c : word) {
            if (!isalpha(c)) continue;
            
            int index = char_to_index(c);
            if (index < 0 || index >= TrieNode::ALPHABET_SIZE) continue;
            
            if (current->children[index] == nullptr) {
                current->children[index] = new TrieNode();
            }
            current = current->children[index];
        }
        
        current->is_end_of_word = true;
        current->occurrences.push_back(info);
    }
    
    bool search(const std::string& word, VerseInfo& result) const {
        TrieNode* current = root;
        
        for (char c : word) {
            if (!isalpha(c)) continue;
            
            int index = char_to_index(c);
            if (index < 0 || index >= TrieNode::ALPHABET_SIZE) continue;
            
            if (current->children[index] == nullptr) {
                return false;
            }
            current = current->children[index];
        }
        
        if (current->is_end_of_word && !current->occurrences.empty()) {
            result = current->occurrences[0];
            return true;
        }
        
        return false;
    }
};

// ==================== TRIE FOR PLAYERS (names) ====================

class TriePlayersNode {
public:
    static const int ALPHABET_SIZE = 26;
    TriePlayersNode* children[ALPHABET_SIZE];
    bool is_end_of_word;
    std::vector<PlayerInfo> occurrences;

    TriePlayersNode() : is_end_of_word(false) {
        for (int i = 0; i < ALPHABET_SIZE; ++i) children[i] = nullptr;
    }
    ~TriePlayersNode() {
        for (int i = 0; i < ALPHABET_SIZE; ++i) if (children[i] != nullptr) delete children[i];
    }
};

class TriePlayers {
private:
    TriePlayersNode* root;
    int char_to_index(char c) const { return tolower(c) - 'a'; }
public:
    TriePlayers() { root = new TriePlayersNode(); }
    ~TriePlayers() { delete root; }

    void insert(const std::string& nameKey, const PlayerInfo& info) {
        TriePlayersNode* current = root;
        for (char c : nameKey) {
            if (!isalpha(c)) continue;
            int index = char_to_index(c);
            if (index < 0 || index >= TriePlayersNode::ALPHABET_SIZE) continue;
            if (current->children[index] == nullptr) current->children[index] = new TriePlayersNode();
            current = current->children[index];
        }
        current->is_end_of_word = true;
        current->occurrences.push_back(info);
    }

    bool search(const std::string& nameKey, PlayerInfo& result) const {
        TriePlayersNode* current = root;
        for (char c : nameKey) {
            if (!isalpha(c)) continue;
            int index = char_to_index(c);
            if (index < 0 || index >= TriePlayersNode::ALPHABET_SIZE) continue;
            if (current->children[index] == nullptr) return false;
            current = current->children[index];
        }
        if (current->is_end_of_word && !current->occurrences.empty()) {
            result = current->occurrences[0];
            return true;
        }
        return false;
    }
};

// ==================== IF-THEN-ELSE IMPLEMENTATION ====================
// This implementation does NOT build any tree structure. Instead, it performs
// an iterative binary search using plain if-else statements over two parallel
// arrays that are already sorted by word. The indexing (collecting and sorting
// words from verses) is performed outside of this class.

class IfThenElseTree {
private:
    // Parallel C arrays (allocated with calloc)
    // words[i] corresponds to infos[i]
    const char** words{nullptr};
    const VerseInfo** infos{nullptr};
    size_t count{0};
    
    std::string to_lower(const std::string& str) const {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
public:
    IfThenElseTree() = default;
    
    ~IfThenElseTree() {}
    
    // Provide pre-sorted parallel arrays to enable binary search.
    void set_index(const char** words_sorted,
                   const VerseInfo** infos_sorted,
                   size_t n) {
        words = words_sorted;
        infos = infos_sorted;
        count = n;
    }
    
    void insert(const std::string& word, const VerseInfo& info) {
        // No-op. Indexing happens outside and is provided via set_index().
    }
    
    bool search(const std::string& word, VerseInfo& result) const {
        // Iterative binary search over the sorted words array.
        if (words == nullptr || infos == nullptr || count == 0) {
            return false;
        }
        
        // Normalize the query to lowercase (words_ref are expected to be lowercase)
        std::string key = to_lower(word);
        
        long left = 0;
        long right = static_cast<long>(count) - 1;
        
        long found = -1;
        while (left <= right) {
            long mid = left + (right - left) / 2;
            const char* midWord = words[static_cast<size_t>(mid)];
            
            int cmp = std::strcmp(midWord, key.c_str());
            if (cmp == 0) {
                // Record and continue to search leftmost occurrence
                found = mid;
                right = mid - 1;
            } else if (cmp < 0) {
                left = mid + 1;
            } else { // midWord > key
                right = mid - 1;
            }
        }
        if (found != -1) {
            result = *infos[static_cast<size_t>(found)];
            return true;
        }
        return false; // Not found
    }
};

// If-then-else binary search for players (on names)

class IfThenElseTreePlayers {
private:
    const char** names{nullptr};
    const PlayerInfo** infos{nullptr};
    size_t count{0};
    static std::string to_lower(const std::string& s) {
        std::string r = s; std::transform(r.begin(), r.end(), r.begin(), ::tolower); return r;
    }
public:
    void set_index(const char** names_sorted,
                   const PlayerInfo** infos_sorted,
                   size_t n) {
        names = names_sorted; infos = infos_sorted; count = n;
    }
    bool search(const std::string& name, PlayerInfo& out) const {
        if (!names || !infos || count == 0) return false;
        std::string key = to_lower(name);
        long l = 0, r = static_cast<long>(count) - 1;
        long found = -1;
        while (l <= r) {
            long m = l + (r - l) / 2;
            const char* mid = names[static_cast<size_t>(m)];
            int cmp = std::strcmp(mid, key.c_str());
            if (cmp == 0) { found = m; r = m - 1; }
            else if (cmp < 0) l = m + 1; else r = m - 1;
        }
        if (found != -1) { out = *infos[static_cast<size_t>(found)]; return true; }
        return false;
    }
};

// ==================== UTILITY FUNCTIONS ====================

std::string clean_word(const std::string& word) {
    std::string result;
    for (char c : word) {
        if (isalpha(c)) {
            result += tolower(c);
        }
    }
    return result;
}

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::string word;
    
    for (char c : text) {
        if (isalpha(c)) {
            word += c;
        } else if (!word.empty()) {
            words.push_back(word);
            word.clear();
        }
    }
    
    if (!word.empty()) {
        words.push_back(word);
    }
    
    return words;
}

// Utilities for players CSV parsing
static inline std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && isspace(static_cast<unsigned char>(s[e-1]))) --e;
    return s.substr(b, e - b);
}

static inline std::string unquote(const std::string& s) {
    if (s.size() >= 2 && ((s.front()=='"' && s.back()=='"') || (s.front()=='\'' && s.back()=='\'')))
        return s.substr(1, s.size()-2);
    return s;
}

static std::vector<std::string> parse_csv_line_3(const std::string& line) {
    // Parse a CSV line honoring quotes and commas inside quotes. Returns fields list.
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];
        if (ch == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                // Escaped quote
                cur.push_back('"');
                ++i; // skip next
            } else {
                in_quotes = !in_quotes;
            }
        } else if (ch == ',' && !in_quotes) {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    fields.push_back(cur);
    return fields;
}

bool load_players_file(const std::string& filename, std::vector<PlayerInfo>& players) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open players file " << filename << std::endl;
        return false;
    }
    std::string line;
    bool header_skipped = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        // Skip header if first column non-digit or contains 'id'
        if (!header_skipped) {
            std::string l = line; std::transform(l.begin(), l.end(), l.begin(), ::tolower);
            if (l.find("id") != std::string::npos && l.find("name") != std::string::npos) {
                header_skipped = true; continue;
            }
            header_skipped = true; // treat first as data if not header-like
        }
        // Robust CSV split honoring quotes
        std::vector<std::string> cols = parse_csv_line_3(line);
        if (cols.size() < 3) continue;
        std::string id = trim(unquote(cols[0]));
        std::string name = trim(unquote(cols[1]));
        // Preserve quotes for positions as present in CSV to reflect exact field (including commas)
        std::string position = trim(cols[2]);
        if (!name.empty()) players.emplace_back(id, name, position);
    }
    return true;
}

// Bible file parser for King James Version format
// Expected format: Lines with "Chapter:Verse Text" with book titles on separate lines
// Example: "1:1 In the beginning God created the heaven and the earth."
bool load_bible_file(const std::string& filename, std::vector<VerseInfo>& verses) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    std::string current_book = "Unknown";
    std::string accumulated_text;
    bool in_verse = false;
    int current_chapter = 0;
    int current_verse = 0;
    
    // Map to detect book titles more reliably
    auto detect_book = [](std::string line) -> std::string {
        // Remove carriage return if present (Windows line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        // Must be a standalone title line (not part of verse text)
        // Check for common patterns at the beginning of the line
        if (line == "The First Book of Moses: Called Genesis") return "Genesis";
        if (line == "The Second Book of Moses: Called Exodus") return "Exodus";
        if (line == "The Third Book of Moses: Called Leviticus") return "Leviticus";
        if (line == "The Fourth Book of Moses: Called Numbers") return "Numbers";
        if (line == "The Fifth Book of Moses: Called Deuteronomy") return "Deuteronomy";
        if (line == "The Book of Joshua") return "Joshua";
        if (line == "The Book of Judges") return "Judges";
        if (line == "The Book of Ruth") return "Ruth";
        if (line == "The First Book of Samuel") return "1Samuel";
        if (line == "The Second Book of Samuel") return "2Samuel";
        if (line == "The First Book of the Kings") return "1Kings";
        if (line == "The Second Book of the Kings") return "2Kings";
        if (line == "The First Book of the Chronicles") return "1Chronicles";
        if (line == "The Second Book of the Chronicles") return "2Chronicles";
        if (line == "Ezra") return "Ezra";
        if (line == "The Book of Nehemiah") return "Nehemiah";
        if (line == "The Book of Esther") return "Esther";
        if (line == "The Book of Job") return "Job";
        if (line == "The Book of Psalms") return "Psalms";
        if (line == "The Proverbs") return "Proverbs";
        if (line == "Ecclesiastes") return "Ecclesiastes";
        if (line == "The Song of Solomon") return "Song_of_Solomon";
        if (line == "The Book of the Prophet Isaiah") return "Isaiah";
        if (line == "The Book of the Prophet Jeremiah") return "Jeremiah";
        if (line == "The Lamentations of Jeremiah") return "Lamentations";
        if (line == "The Book of the Prophet Ezekiel") return "Ezekiel";
        if (line == "The Book of Daniel") return "Daniel";
        if (line == "Hosea") return "Hosea";
        if (line == "Joel") return "Joel";
        if (line == "Amos") return "Amos";
        if (line == "Obadiah") return "Obadiah";
        if (line == "Jonah") return "Jonah";
        if (line == "Micah") return "Micah";
        if (line == "Nahum") return "Nahum";
        if (line == "Habakkuk") return "Habakkuk";
        if (line == "Zephaniah") return "Zephaniah";
        if (line == "Haggai") return "Haggai";
        if (line == "Zechariah") return "Zechariah";
        if (line == "Malachi") return "Malachi";
        if (line == "The Gospel According to Saint Matthew") return "Matthew";
        if (line == "The Gospel According to Saint Mark") return "Mark";
        if (line == "The Gospel According to Saint Luke") return "Luke";
        if (line == "The Gospel According to Saint John") return "John";
        if (line == "The Acts of the Apostles") return "Acts";
        if (line == "The Epistle of Paul the Apostle to the Romans") return "Romans";
        if (line == "The First Epistle of Paul the Apostle to the Corinthians") return "1Corinthians";
        if (line == "The Second Epistle of Paul the Apostle to the Corinthians") return "2Corinthians";
        if (line == "The Epistle of Paul the Apostle to the Galatians") return "Galatians";
        if (line == "The Epistle of Paul the Apostle to the Ephesians") return "Ephesians";
        if (line == "The Epistle of Paul the Apostle to the Philippians") return "Philippians";
        if (line == "The Epistle of Paul the Apostle to the Colossians") return "Colossians";
        if (line == "The First Epistle of Paul the Apostle to the Thessalonians") return "1Thessalonians";
        if (line == "The Second Epistle of Paul the Apostle to the Thessalonians") return "2Thessalonians";
        if (line == "The First Epistle of Paul the Apostle to Timothy") return "1Timothy";
        if (line == "The Second Epistle of Paul the Apostle to Timothy") return "2Timothy";
        if (line == "The Epistle of Paul the Apostle to Titus") return "Titus";
        if (line == "The Epistle of Paul the Apostle to Philemon") return "Philemon";
        if (line == "The Epistle of Paul the Apostle to the Hebrews") return "Hebrews";
        if (line == "The General Epistle of James") return "James";
        if (line == "The First Epistle General of Peter") return "1Peter";
        if (line == "The Second General Epistle of Peter") return "2Peter";
        if (line == "The First Epistle General of John") return "1John";
        if (line == "The Second Epistle General of John") return "2John";
        if (line == "The Third Epistle General of John") return "3John";
        if (line == "The General Epistle of Jude") return "Jude";
        if (line == "The Revelation of Saint John the Divine") return "Revelation";
        
        return "";
    };
    
    while (std::getline(file, line)) {
        // Skip Project Gutenberg header and empty lines at the beginning
        if (line.find("PROJECT GUTENBERG") != std::string::npos ||
            line.find("***") != std::string::npos ||
            line == "The Old Testament of the King James Version of the Bible" ||
            line == "The New Testament of the King James Bible") {
            continue;
        }
        
        // Try to detect book title
        std::string detected_book = detect_book(line);
        if (!detected_book.empty()) {
            // Save accumulated verse if any
            if (in_verse && !accumulated_text.empty()) {
                verses.push_back(VerseInfo(current_book, current_chapter, current_verse, accumulated_text));
                accumulated_text.clear();
                in_verse = false;
            }
            current_book = detected_book;
            continue;
        }
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Check if line starts with chapter:verse pattern
        if (isdigit(line[0])) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos && colon_pos < 10) {
                // Check if this looks like a verse reference (not just any colon)
                bool is_verse_ref = true;
                for (size_t i = 0; i < colon_pos; ++i) {
                    if (!isdigit(line[i])) {
                        is_verse_ref = false;
                        break;
                    }
                }
                
                if (is_verse_ref) {
                    // Save previous verse if any
                    if (in_verse && !accumulated_text.empty()) {
                        verses.push_back(VerseInfo(current_book, current_chapter, current_verse, accumulated_text));
                        accumulated_text.clear();
                    }
                    
                    // Parse new verse
                    size_t space_pos = line.find(' ', colon_pos);
                    if (space_pos != std::string::npos) {
                        try {
                            current_chapter = std::stoi(line.substr(0, colon_pos));
                            current_verse = std::stoi(line.substr(colon_pos + 1, space_pos - colon_pos - 1));
                            accumulated_text = line.substr(space_pos + 1);
                            
                            // Remove carriage return at the end if present
                            if (!accumulated_text.empty() && accumulated_text.back() == '\r') {
                                accumulated_text.pop_back();
                            }
                            
                            in_verse = true;
                        } catch (...) {
                            continue;
                        }
                    }
                } else if (in_verse) {
                    // Line starts with digit but isn't a verse reference
                    if (!accumulated_text.empty()) accumulated_text += " ";
                    
                    // Remove carriage return at the end if present
                    std::string clean_line = line;
                    if (!clean_line.empty() && clean_line.back() == '\r') {
                        clean_line.pop_back();
                    }
                    accumulated_text += clean_line;
                }
            } else if (in_verse) {
                // Continuation of previous verse (line starts with digit but no colon nearby)
                if (!accumulated_text.empty()) accumulated_text += " ";
                
                // Remove carriage return at the end if present
                std::string clean_line = line;
                if (!clean_line.empty() && clean_line.back() == '\r') {
                    clean_line.pop_back();
                }
                accumulated_text += clean_line;
            }
        } else if (in_verse) {
            // Continuation of verse text
            if (!accumulated_text.empty()) accumulated_text += " ";
            
            // Remove carriage return at the end if present
            std::string clean_line = line;
            if (!clean_line.empty() && clean_line.back() == '\r') {
                clean_line.pop_back();
            }
            accumulated_text += clean_line;
        }
    }
    
    // Save last verse if any
    if (in_verse && !accumulated_text.empty()) {
        verses.push_back(VerseInfo(current_book, current_chapter, current_verse, accumulated_text));
    }
    
    file.close();
    return true;
}

// ==================== MAIN PROGRAM ====================

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset_file> [dataset_type=bible] <structure_type> <search_word> [<search_word> ...]\n";
        std::cerr << "dataset_type: bible | players (default: bible)\nstructure_type: trie | ifthenelse" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::string dataset_type;
    std::string structure;
    // Support multiple search words (interleaved existing/nonexisting)
    std::vector<std::string> queries;
    if (argc >= 5) {
        dataset_type = argv[2];
        structure = argv[3];
        for (int i = 4; i < argc; ++i) {
            queries.emplace_back(argv[i]);
        }
    } else {
        dataset_type = "bible";
        structure = argv[2];
        queries.emplace_back(argv[3]);
    }
    
    std::cout << "Dataset type: " << dataset_type << std::endl;
    std::cout << "Loading dataset from: " << filename << std::endl;
    
    if (dataset_type == "bible") {
        std::vector<VerseInfo> verses;
        if (!load_bible_file(filename, verses)) {
            return 1;
        }
        std::cout << "Loaded " << verses.size() << " verses." << std::endl;
        std::cout << "Building " << structure << " structure..." << std::endl;

        if (structure == "trie") {
        Trie trie;
        // Timings: no sorting for trie
        auto t_build_start = std::chrono::steady_clock::now();
        
        // Index all words
        for (const auto& verse : verses) {
            std::vector<std::string> words = split_words(verse.text);
            for (const auto& word : words) {
                std::string clean = clean_word(word);
                if (!clean.empty()) {
                    trie.insert(clean, verse);
                }
            }
        }
        auto t_build_end = std::chrono::steady_clock::now();
        
        std::cout << "Searching for " << queries.size() << " words (interleaved existing/nonexisting)..." << std::endl;
        auto t_search_start = std::chrono::steady_clock::now();
        size_t found_cnt = 0;
        for (const auto& qraw : queries) {
            std::string q = clean_word(qraw);
            if (q.empty()) continue;
            VerseInfo result;
            if (trie.search(q, result)) {
                ++found_cnt;
            }
        }
        auto t_search_end = std::chrono::steady_clock::now();
        std::cout << "Search results: " << found_cnt << "/" << queries.size() << " found." << std::endl;

        // Report timing breakdown (ms)
        double sort_ms = 0.0;
        double build_ms = std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count();
        double search_ms = std::chrono::duration<double, std::milli>(t_search_end - t_search_start).count();
        double total_ms = sort_ms + build_ms + search_ms;
        std::cout << "\nTIMING_MS sort=" << sort_ms
                  << " build=" << build_ms
                  << " search=" << search_ms
                  << " total=" << total_ms << std::endl;
        
    } else if (structure == "ifthenelse") {
        // Build a flat index of words -> verse occurrences.
        std::vector<std::string> words_index;
        std::vector<VerseInfo> infos_index;
        words_index.reserve(verses.size() * 5); // rough heuristic to reduce reallocations
        infos_index.reserve(verses.size() * 5);

        auto t_build_collect_start = std::chrono::steady_clock::now();
        for (const auto& verse : verses) {
            std::vector<std::string> words = split_words(verse.text);
            for (const auto& w : words) {
                std::string clean = clean_word(w); // lowercase alpha-only
                if (!clean.empty()) {
                    words_index.push_back(clean);
                    infos_index.push_back(verse);
                }
            }
        }
        auto t_build_collect_end = std::chrono::steady_clock::now();

        // Sort the index by word using an order vector (to keep arrays parallel)
        std::vector<size_t> order(words_index.size());
        std::iota(order.begin(), order.end(), 0);
        auto t_sort_start = std::chrono::steady_clock::now();
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b){
            return words_index[a] < words_index[b];
        });
        
        auto t_sort_end = std::chrono::steady_clock::now();

        // Build C arrays allocated with calloc
        size_t n = order.size();
        const char** words_arr = (const char**)std::calloc(n, sizeof(char*));
        const VerseInfo** infos_arr = (const VerseInfo**)std::calloc(n, sizeof(VerseInfo*));
        auto t_build_struct_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n; ++i) {
            size_t idx = order[i];
            const std::string& w = words_index[idx];
            char* wcopy = (char*)std::calloc(w.size() + 1, sizeof(char));
            std::memcpy(wcopy, w.c_str(), w.size());
            words_arr[i] = wcopy;
            infos_arr[i] = &infos_index[idx];
        }
        auto t_build_struct_end = std::chrono::steady_clock::now();

        IfThenElseTree tree;
        tree.set_index(words_arr, infos_arr, n);

        std::cout << "Searching for " << queries.size() << " words (interleaved existing/nonexisting)..." << std::endl;
        auto t_search_start = std::chrono::steady_clock::now();
        size_t found_cnt = 0;
        for (const auto& qraw : queries) {
            std::string query = clean_word(qraw);
            if (query.empty()) continue;
            VerseInfo result;
            if (tree.search(query, result)) {
                ++found_cnt;
            }
        }
        auto t_search_end = std::chrono::steady_clock::now();
        std::cout << "Search results: " << found_cnt << "/" << queries.size() << " found." << std::endl;

        // Report timing breakdown (ms)
        double sort_ms = std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count();
        double build_ms = std::chrono::duration<double, std::milli>(t_build_collect_end - t_build_collect_start).count()
                        + std::chrono::duration<double, std::milli>(t_build_struct_end - t_build_struct_start).count();
        double search_ms = std::chrono::duration<double, std::milli>(t_search_end - t_search_start).count();
        double total_ms = sort_ms + build_ms + search_ms;
        std::cout << "\nTIMING_MS sort=" << sort_ms
                  << " build=" << build_ms
                  << " search=" << search_ms
                  << " total=" << total_ms << std::endl;
        // Free allocated C arrays
        for (size_t i = 0; i < n; ++i) { std::free((void*)words_arr[i]); }
        std::free((void*)words_arr);
        std::free((void*)infos_arr);
        }
    } else if (dataset_type == "players") {
        std::vector<PlayerInfo> players;
        if (!load_players_file(filename, players)) {
            return 1;
        }
        std::cout << "Loaded " << players.size() << " players." << std::endl;
        std::cout << "Building " << structure << " structure..." << std::endl;

        if (structure == "trie") {
            TriePlayers trie;
            auto t_build_start = std::chrono::steady_clock::now();
            for (const auto& p : players) {
                std::string key = clean_word(p.name);
                if (!key.empty()) trie.insert(key, p);
            }
            auto t_build_end = std::chrono::steady_clock::now();
            std::cout << "Searching for " << queries.size() << " players (interleaved existing/nonexisting)..." << std::endl;
            size_t found_cnt = 0;
            auto t_search_start = std::chrono::steady_clock::now();
            for (const auto& qraw : queries) {
                std::string q = clean_word(qraw);
                if (q.empty()) continue;
                PlayerInfo out;
                if (trie.search(q, out)) {
                    ++found_cnt;
                }
            }
            auto t_search_end = std::chrono::steady_clock::now();
            std::cout << "Search results: " << found_cnt << "/" << queries.size() << " found." << std::endl;
            double sort_ms = 0.0;
            double build_ms = std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count();
            double search_ms = std::chrono::duration<double, std::milli>(t_search_end - t_search_start).count();
            double total_ms = sort_ms + build_ms + search_ms;
            std::cout << "\nTIMING_MS sort=" << sort_ms
                      << " build=" << build_ms
                      << " search=" << search_ms
                      << " total=" << total_ms << std::endl;
        } else if (structure == "ifthenelse") {
            std::vector<std::string> names_index;
            std::vector<PlayerInfo> infos_index;
            names_index.reserve(players.size());
            infos_index.reserve(players.size());
            auto t_build_collect_start = std::chrono::steady_clock::now();
            for (const auto& p : players) {
                std::string key = clean_word(p.name);
                if (!key.empty()) { names_index.push_back(key); infos_index.push_back(p); }
            }
            auto t_build_collect_end = std::chrono::steady_clock::now();
            std::vector<size_t> order(names_index.size());
            std::iota(order.begin(), order.end(), 0);
            auto t_sort_start = std::chrono::steady_clock::now();
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b){ return names_index[a] < names_index[b]; });
            auto t_sort_end = std::chrono::steady_clock::now();
            // Build C arrays allocated with calloc
            size_t n = order.size();
            const char** names_arr = (const char**)std::calloc(n, sizeof(char*));
            const PlayerInfo** infos_arr = (const PlayerInfo**)std::calloc(n, sizeof(PlayerInfo*));
            auto t_build_struct_start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < n; ++i) {
                size_t idx = order[i];
                const std::string& w = names_index[idx];
                char* wcopy = (char*)std::calloc(w.size() + 1, sizeof(char));
                std::memcpy(wcopy, w.c_str(), w.size());
                names_arr[i] = wcopy;
                infos_arr[i] = &infos_index[idx];
            }
            auto t_build_struct_end = std::chrono::steady_clock::now();

            IfThenElseTreePlayers tree;
            tree.set_index(names_arr, infos_arr, n);
            std::cout << "Searching for " << queries.size() << " players (interleaved existing/nonexisting)..." << std::endl;
            size_t found_cnt = 0;
            auto t_search_start = std::chrono::steady_clock::now();
            for (const auto& qraw : queries) {
                std::string q = clean_word(qraw);
                if (q.empty()) continue;
                PlayerInfo out;
                if (tree.search(q, out)) {
                    ++found_cnt;
                }
            }
            auto t_search_end = std::chrono::steady_clock::now();
            std::cout << "Search results: " << found_cnt << "/" << queries.size() << " found." << std::endl;
            double sort_ms = std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count();
            double build_ms = std::chrono::duration<double, std::milli>(t_build_collect_end - t_build_collect_start).count()
                            + std::chrono::duration<double, std::milli>(t_build_struct_end - t_build_struct_start).count();
            double search_ms = std::chrono::duration<double, std::milli>(t_search_end - t_search_start).count();
            double total_ms = sort_ms + build_ms + search_ms;
            std::cout << "\nTIMING_MS sort=" << sort_ms
                      << " build=" << build_ms
                      << " search=" << search_ms
                      << " total=" << total_ms << std::endl;
            // Free allocated C arrays
            for (size_t i = 0; i < n; ++i) { std::free((void*)names_arr[i]); }
            std::free((void*)names_arr);
            std::free((void*)infos_arr);
        } else {
            std::cerr << "Unknown structure type: " << structure << std::endl;
            std::cerr << "Use 'trie' or 'ifthenelse'" << std::endl;
            return 2;
        }
    } else {
        std::cerr << "Unknown structure type: " << structure << std::endl;
        std::cerr << "Use 'trie' or 'ifthenelse'" << std::endl;
        return 2;
    }
    
    return 0;
}
