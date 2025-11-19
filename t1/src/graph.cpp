// Consolidated and minimal includes
#include <cstddef>
#include <cctype>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>

// Interface for generic graph
class IGraph {
public:
    virtual void add_edge(size_t u, size_t v) = 0;
    virtual bool has_edge(size_t u, size_t v) const = 0;
    virtual size_t degree(size_t u) const = 0;
    virtual size_t get_size() const = 0;
    virtual void add_nodes(size_t count) = 0;
    virtual ~IGraph() {}
};

// Adjacency matrix using native boolean arrays with calloc
class MatrixGraph : public IGraph {
private:
    size_t size;
    bool** adj_matrix;
public:
    explicit MatrixGraph(size_t n) : size(n) {
        adj_matrix = (bool**)calloc(size, sizeof(bool*));
        for (size_t i = 0; i < size; ++i) {
            adj_matrix[i] = (bool*)calloc(size, sizeof(bool));
        }
    }
    ~MatrixGraph() override {
        for (size_t i = 0; i < size; ++i) free(adj_matrix[i]);
        free(adj_matrix);
    }
    void add_edge(size_t u, size_t v) override {
        if (u < size && v < size) {
            adj_matrix[u][v] = true;
            adj_matrix[v][u] = true;
        }
    }
    bool has_edge(size_t u, size_t v) const override {
        if (u < size && v < size) return adj_matrix[u][v];
        return false;
    }
    size_t degree(size_t u) const override {
        if (u < size) {
            size_t deg = 0;
            for (size_t v = 0; v < size; ++v) if (adj_matrix[u][v]) ++deg;
            return deg;
        }
        return 0;
    }
    size_t get_size() const override { return size; }
    void add_nodes(size_t count) override {
        if (count == 0) return;
        size_t new_size = size + count;
        bool** new_matrix = (bool**)calloc(new_size, sizeof(bool*));
        for (size_t i = 0; i < new_size; ++i) {
            new_matrix[i] = (bool*)calloc(new_size, sizeof(bool));
        }
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                new_matrix[i][j] = adj_matrix[i][j];
            }
        }
        for (size_t i = 0; i < size; ++i) free(adj_matrix[i]);
        free(adj_matrix);
        adj_matrix = new_matrix;
        size = new_size;
    }
};

// Adjacency list using per-vertex dynamic arrays calloc-based
class ListGraph : public IGraph {
private:
    size_t size;
    size_t* degrees;    // current degree per vertex
    size_t* capacities; // current capacity per vertex
    size_t** neighbors; // adjacency lists
public:
    explicit ListGraph(size_t n) : size(n) {
        // Arrays zero-initialized by calloc
        // current degree per vertex
        degrees = (size_t*)calloc(size, sizeof(size_t));
        // current capacity per vertex
        capacities = (size_t*)calloc(size, sizeof(size_t));
        // adjacency lists (nullptr initially)
        neighbors = (size_t**)calloc(size, sizeof(size_t*));
    }
    ~ListGraph() override {
        if (!neighbors) return;
        for (size_t i = 0; i < size; ++i) {
            if (neighbors[i]) free(neighbors[i]);
        }
        free(neighbors);
        free(degrees);
        free(capacities);
    }
    void add_edge(size_t u, size_t v) override {
        if (u >= size || v >= size) return;
        ensure_capacity(u, degrees[u] + 1);
        ensure_capacity(v, degrees[v] + 1);
        neighbors[u][degrees[u]++] = v;
        neighbors[v][degrees[v]++] = u;
    }
    bool has_edge(size_t u, size_t v) const override {
        if (u >= size || v >= size) return false;
        size_t deg = degrees[u];
        const size_t* nbrs = neighbors[u];
        for (size_t i = 0; i < deg; ++i) {
            if (nbrs[i] == v) return true;
        }
        return false;
    }
    size_t degree(size_t u) const override {
        if (u < size) return degrees[u];
        return 0;
    }
    size_t get_size() const override { return size; }
    void add_nodes(size_t count) override {
        if (count == 0) return;
        size_t old_size = size;
        size_t new_size = size + count;

        // Allocate new arrays with calloc and copy existing data
        size_t* new_degrees = (size_t*)calloc(new_size, sizeof(size_t));
        size_t* new_capacities = (size_t*)calloc(new_size, sizeof(size_t));
        size_t** new_neighbors = (size_t**)calloc(new_size, sizeof(size_t*));
        if (!new_degrees || !new_capacities || !new_neighbors) {
            // Cleanup partially allocated arrays to avoid leaks on OOM
            if (new_degrees) free(new_degrees);
            if (new_capacities) free(new_capacities);
            if (new_neighbors) free(new_neighbors);
            return;
        }
        // Copy old metadata and pointers
        if (degrees) std::memcpy(new_degrees, degrees, old_size * sizeof(size_t));
        if (capacities) std::memcpy(new_capacities, capacities, old_size * sizeof(size_t));
        if (neighbors) std::memcpy(new_neighbors, neighbors, old_size * sizeof(size_t*));

        free(degrees);
        free(capacities);
        free(neighbors);
        degrees = new_degrees;
        capacities = new_capacities;
        neighbors = new_neighbors;
        size = new_size;
    }
private:
    // Ensure neighbors[u] has at least min_capacity slots
    void ensure_capacity(size_t u, size_t min_required) {
        size_t cap = capacities[u];
        if (cap == 0) {
            size_t init_cap = (min_required > 4) ? min_required : 4;
            neighbors[u] = (size_t*)calloc(init_cap, sizeof(size_t));
            if (!neighbors[u]) return;
            capacities[u] = init_cap;
            return;
        }
        if (min_required <= cap) return;
        size_t new_cap = cap;
        while (new_cap < min_required) new_cap <<= 1; // exponential growth
        size_t* new_arr = (size_t*)calloc(new_cap, sizeof(size_t));
        if (!new_arr) return;
        // copy existing neighbors
        std::memcpy(new_arr, neighbors[u], degrees[u] * sizeof(size_t));
        free(neighbors[u]);
        neighbors[u] = new_arr;
        capacities[u] = new_cap;
    }
};

// Adjacency hash (custom open-addressing hash table)
class HashGraph : public IGraph {
private:
    struct NeighborSet {
        size_t* keys;     // buckets
        size_t capacity;  // power of two, 0 when empty
        size_t count;     // number of elements
    };
    static constexpr double LOAD_FACTOR = 0.75;
    static constexpr size_t EMPTY = (size_t)(~(size_t)0); // SIZE_MAX
    size_t size;
    NeighborSet* adj_sets;
public:
    explicit HashGraph(size_t n) : size(n) {
        // One neighbor set per vertex
        adj_sets = (NeighborSet*)calloc(size, sizeof(NeighborSet));
        // NeighborSet is zeroed by calloc
        // (count=0, capacity=0, keys=nullptr)
    }
    ~HashGraph() override {
        if (!adj_sets) return;
        for (size_t i = 0; i < size; ++i) {
            if (adj_sets[i].keys) free(adj_sets[i].keys);
        }
        free(adj_sets);
    }
    void add_edge(size_t u, size_t v) override {
        if (u < size && v < size) {
            insert(adj_sets[u], v);
            insert(adj_sets[v], u);
        }
    }
    bool has_edge(size_t u, size_t v) const override {
        if (u < size && v < size) {
            return contains(adj_sets[u], v);
        }
        return false;
    }
    size_t degree(size_t u) const override {
        if (u < size) return adj_sets[u].count;
        return 0;
    }
    size_t get_size() const override { return size; }
    void add_nodes(size_t count) override {
        if (count == 0) return;
        size_t old_size = size;
        size_t new_size = size + count;
        NeighborSet* new_sets = (NeighborSet*)realloc(adj_sets, new_size * sizeof(NeighborSet));
        if (!new_sets) return;
        adj_sets = new_sets;
        for (size_t i = old_size; i < new_size; ++i) {
            adj_sets[i].keys = nullptr;
            adj_sets[i].capacity = 0;
            adj_sets[i].count = 0;
        }
        size = new_size;
    }
private:
    static inline size_t mix_hash(size_t x) {
        // Fibonacci hashing for good dispersion
        // masked with cap-1 by caller
        return (size_t)(x * 11400714819323198485ull);
    }
    static inline size_t next_pow2(size_t x) {
        if (x < 8) return 8;
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
        x |= x >> 32;
#endif
        return x + 1;
    }
    static void init_keys(size_t* keys, size_t cap) {
        // Fill all buckets with EMPTY
        for (size_t i = 0; i < cap; ++i) keys[i] = EMPTY;
    }
    static bool contains(const NeighborSet& set, size_t key) {
        if (set.capacity == 0) return false;
        size_t mask = set.capacity - 1;
        size_t idx = mix_hash(key) & mask;
        // Linear probing until EMPTY (not found) or the key is matched
        while (true) {
            size_t k = set.keys[idx];
            if (k == EMPTY) return false;
            if (k == key) return true;
            idx = (idx + 1) & mask;
        }
    }
    static void rehash(NeighborSet& set, size_t new_cap) {
        size_t* old_keys = set.keys;
        size_t old_cap = set.capacity;
        set.keys = (size_t*)malloc(new_cap * sizeof(size_t));
        set.capacity = new_cap;
        set.count = 0;
        init_keys(set.keys, new_cap);
        if (old_keys) {
            for (size_t i = 0; i < old_cap; ++i) {
                size_t k = old_keys[i];
                if (k != EMPTY) {
                    size_t mask = new_cap - 1;
                    size_t idx = mix_hash(k) & mask;
                    while (set.keys[idx] != EMPTY) {
                        idx = (idx + 1) & mask;
                    }
                    set.keys[idx] = k;
                    ++set.count;
                }
            }
            free(old_keys);
        }
    }
    static void ensure_capacity(NeighborSet& set, size_t min_needed) {
        size_t cap = set.capacity;
        if (cap == 0) {
            size_t init_cap = next_pow2(min_needed);
            rehash(set, init_cap);
            return;
        }
        // If count+1 exceeds the load factor, double the capacity
        if ((double)(set.count + 1) > LOAD_FACTOR * (double)cap) {
            rehash(set, cap * 2);
        }
    }
    static void insert(NeighborSet& set, size_t key) {
        ensure_capacity(set, 8);
        size_t mask = set.capacity - 1;
        size_t idx = mix_hash(key) & mask;
        while (true) {
            size_t k = set.keys[idx];
            if (k == EMPTY) {
                set.keys[idx] = key;
                ++set.count;
                return;
            }
            if (k == key) {
                return; // already present
            }
            idx = (idx + 1) & mask;
        }
    }
};

// Naive adjacency matrix using std::vector (row-major char matrix)
class VectorMatrixGraph : public IGraph {
private:
    size_t size;
    std::vector<std::vector<char>> adj;
public:
    explicit VectorMatrixGraph(size_t n) : size(n), adj(n, std::vector<char>(n, 0)) {}
    ~VectorMatrixGraph() override = default;
    void add_edge(size_t u, size_t v) override {
        if (u < size && v < size) {
            adj[u][v] = 1;
            adj[v][u] = 1;
        }
    }
    bool has_edge(size_t u, size_t v) const override {
        if (u < size && v < size) return adj[u][v] != 0;
        return false;
    }
    size_t degree(size_t u) const override {
        if (u >= size) return 0;
        size_t deg = 0;
        const auto &row = adj[u];
        for (size_t v = 0; v < size; ++v) if (row[v]) ++deg;
        return deg;
    }
    size_t get_size() const override { return size; }
    void add_nodes(size_t count) override {
        if (count == 0) return;
        size_t new_size = size + count;
        // Resize existing rows
        for (size_t i = 0; i < size; ++i) {
            adj[i].resize(new_size, 0);
        }
        // Add new rows
        adj.resize(new_size, std::vector<char>(new_size, 0));
        size = new_size;
    }
};

// Naive adjacency list using std::vector per vertex
class VectorListGraph : public IGraph {
private:
    size_t size;
    std::vector<std::vector<size_t>> adj;
public:
    explicit VectorListGraph(size_t n) : size(n), adj(n) {}
    ~VectorListGraph() override = default;
    void add_edge(size_t u, size_t v) override {
        if (u >= size || v >= size) return;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    bool has_edge(size_t u, size_t v) const override {
        if (u >= size || v >= size) return false;
        const auto &nbrs = adj[u];
        for (size_t x : nbrs) if (x == v) return true;
        return false;
    }
    size_t degree(size_t u) const override {
        if (u >= size) return 0;
        return adj[u].size();
    }
    size_t get_size() const override { return size; }
    void add_nodes(size_t count) override {
        if (count == 0) return;
        size += count;
        adj.resize(size);
    }
};

// Naive adjacency hash using std::unordered_map
// per vertex (neighbor -> 1)
class UMapGraph : public IGraph {
private:
    size_t size;
    std::vector<std::unordered_map<size_t, char>> adj;
public:
    explicit UMapGraph(size_t n) : size(n), adj(n) {}
    ~UMapGraph() override = default;
    void add_edge(size_t u, size_t v) override {
        if (u >= size || v >= size) return;
        adj[u][v] = 1;
        adj[v][u] = 1;
    }
    bool has_edge(size_t u, size_t v) const override {
        if (u >= size || v >= size) return false;
        auto it = adj[u].find(v);
        return it != adj[u].end();
    }
    size_t degree(size_t u) const override {
        if (u >= size) return 0;
        return adj[u].size();
    }
    size_t get_size() const override { return size; }
    void add_nodes(size_t count) override {
        if (count == 0) return;
        size += count;
        adj.resize(size);
    }
};

#define MAX_NODES 210000

// Reads the dataset and returns the number of nodes and the edge list
struct Edge { size_t u, v; };

size_t read_dataset(const std::string& filename, Edge*& edges, size_t& num_edges) {
    std::ifstream infile(filename.c_str());
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        std::exit(1);
    }
    std::string line;
    size_t max_node = 0;
    size_t edge_count = 0;
    size_t edge_capacity = 1000000; // initial guess, will grow if needed
    edges = new Edge[edge_capacity];
    bool is_csv = filename.size() >= 4 && filename.substr(filename.size() - 4) == ".csv";
    bool first_line = true;

    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;

        if (is_csv && first_line) {
            // Skip header if it contains non-digit characters
            bool has_alpha = false;
            for (char c : line) {
                if (std::isalpha(static_cast<unsigned char>(c))) { has_alpha = true; break; }
            }
            if (has_alpha) { first_line = false; continue; }
        }
        first_line = false;

        size_t u, v;
        bool valid = false;
        if (is_csv) {
            // CSV line format: u,v
            size_t comma = line.find(',');
            if (comma == std::string::npos) continue;
            try {
                u = std::stoull(line.substr(0, comma));
                v = std::stoull(line.substr(comma + 1));
                valid = true;
            } catch (...) {
                continue;
            }
        } else {
            std::istringstream iss(line);
            std::string token1, token2;
            if (!(iss >> token1 >> token2)) continue;
            try {
                u = std::stoull(token1);
                v = std::stoull(token2);
                valid = true;
            } catch (...) {
                continue;
            }
        }
        if (!valid) continue;
        if (u >= MAX_NODES || v >= MAX_NODES) continue;

        if (edge_count == edge_capacity) {
            edge_capacity *= 2;
            Edge* new_edges = new Edge[edge_capacity];
            for (size_t i = 0; i < edge_count; ++i) new_edges[i] = edges[i];
            delete[] edges;
            edges = new_edges;
        }
        edges[edge_count++] = {u, v};
        if (u > max_node) max_node = u;
        if (v > max_node) max_node = v;
    }
    infile.close();
    num_edges = edge_count;
    return max_node + 1; // number of nodes
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset.txt|.csv> <structure_type> <operation>\n";
        std::cerr << "structure_type: matrix | list | hash | matrix_vec | list_vec | hash_umap" << std::endl;
        std::cerr << "operation: insertion | components | clustering" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::string structure = argv[2];
    std::string operation = argv[3];

    Edge* edges = nullptr;
    size_t num_edges = 0;

    std::cout << "Reading dataset..." << std::endl;
    size_t num_nodes = read_dataset(filename, edges, num_edges);
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "Total edges: " << num_edges << std::endl;

    IGraph* graph = nullptr;
    if (structure == "matrix") {
        graph = new MatrixGraph(num_nodes);
    } else if (structure == "list") {
        graph = new ListGraph(num_nodes);
    } else if (structure == "hash") {
        graph = new HashGraph(num_nodes);
    } else if (structure == "matrix_vec") {
        graph = new VectorMatrixGraph(num_nodes);
    } else if (structure == "list_vec") {
        graph = new VectorListGraph(num_nodes);
    } else if (structure == "hash_umap") {
        graph = new UMapGraph(num_nodes);
    } else {
        std::cerr << "Unknown structure type: " << structure << std::endl;
        delete[] edges;
        return 2;
    }

    for (size_t i = 0; i < num_edges; ++i) {
        graph->add_edge(edges[i].u, edges[i].v);
    }
    std::cout << "Graph built with " << graph->get_size() << " nodes." << std::endl;

    if (operation == "insertion") {
        std::cout << "\n--- Node Insertion ---" << std::endl;
        size_t old_size = graph->get_size();
        graph->add_nodes(15);
        for (size_t i = 0; i < 5; ++i) graph->add_edge(0, old_size + i);                 // beginning
        for (size_t i = 0; i < 5; ++i) graph->add_edge(old_size / 2, old_size + 5 + i);  // middle
        for (size_t i = 0; i < 5; ++i) graph->add_edge(old_size - 1, old_size + 10 + i); // end
        std::cout << "Inserted 15 nodes (5 at start, 5 at middle, 5 at end)." << std::endl;
    } else if (operation == "components") {
        std::cout << "\n--- Connected Components ---" << std::endl;
        std::vector<bool> visited(graph->get_size(), false);
        size_t components = 0;
        for (size_t u = 0; u < graph->get_size(); ++u) {
            if (!visited[u]) {
                // BFS
                std::vector<size_t> queue;
                queue.push_back(u);
                visited[u] = true;
                while (!queue.empty()) {
                    size_t curr = queue.back(); queue.pop_back();
                    for (size_t v = 0; v < graph->get_size(); ++v) {
                        if (graph->has_edge(curr, v) && !visited[v]) {
                            visited[v] = true;
                            queue.push_back(v);
                        }
                    }
                }
                ++components;
            }
        }
        std::cout << "Number of connected components: " << components << std::endl;
    } else if (operation == "clustering") {
        std::cout << "\n--- Clustering Coefficient ---" << std::endl;
        double total_coeff = 0.0;
        for (size_t u = 0; u < graph->get_size(); ++u) {
            size_t deg = graph->degree(u);
            if (deg < 2) continue;
            size_t links = 0;

            // Count pairs of connected neighbors
            std::vector<size_t> neighbors;
            neighbors.reserve(deg);
            for (size_t v = 0; v < graph->get_size(); ++v) {
                if (graph->has_edge(u, v)) neighbors.push_back(v);
            }
            for (size_t i = 0; i < neighbors.size(); ++i) {
                for (size_t j = i + 1; j < neighbors.size(); ++j) {
                    if (graph->has_edge(neighbors[i], neighbors[j])) ++links;
                }
            }
            double coeff = (2.0 * links) / (deg * (deg - 1));
            total_coeff += coeff;
        }
        double avg_coeff = total_coeff / graph->get_size();
        std::cout << "Average clustering coefficient: " << avg_coeff << std::endl;
    } else {
        std::cout << "Unknown operation: " << operation << std::endl;
    }

    std::cout << "\nDone." << std::endl;
    delete graph;
    delete[] edges;
    return 0;
}
