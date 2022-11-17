#include <vector>
#include <algorithm>
#include <iostream>

void dfs(const std::vector<std::vector<int>>& graph, std::vector<int>& nodes, int cur, int color) {
  nodes[cur] = color;
  for (int i = 0; i < graph.size(); i++) {
    if (graph[cur][i] == 1 && nodes[i] == -1) {
      dfs(graph, nodes, i, color);
    }
  }
}

int color_graph(const std::vector<std::vector<int>>& graph, std::vector<int>& colors) {
  int color = 0;
  for (int i = 0; i < colors.size(); i++) {
    if (colors[i] == -1) {
      dfs(graph, colors, i, color);
      color += 1;
    }
  }
  return color;
} 

bool is_same_color(int node, const std::vector<int>& colors, const std::vector<int>& init) {
  for (int i = 0; i < init.size(); i++) {
    if (init[i] != node && colors[node] == colors[init[i]]) {
      return true;
    }
  }
  return false;
}

bool comp(int x, int y) {
  return (x < y);
}


// [1,0,0,0,1,0,0,0,0,0,1],
// [0,1,0,1,0,0,0,0,0,0,0],
// [0,0,1,0,0,0,0,1,0,0,0],
// [0,1,0,1,0,1,0,0,0,0,0],
// [1,0,0,0,1,0,0,0,0,0,0],
// [0,0,0,1,0,1,0,0,1,1,0],
// [0,0,0,0,0,0,1,1,0,0,0],
// [0,0,1,0,0,0,1,1,0,0,0],
// [0,0,0,0,0,1,0,0,1,0,0],
// [0,0,0,0,0,1,0,0,0,1,0],
// [1,0,0,0,0,0,0,0,0,0,1]

int minMalwareSpread(std::vector<std::vector<int>>& graph, std::vector<int>& initial) {
  std::vector<int> colors(graph.size(), -1);

  int color_num = color_graph(graph, colors);
  std::vector<int> color_count(color_num, 0);
  for (int i = 0; i < colors.size(); i++) {
    color_count[colors[i]] += 1;
  }

  std::sort(initial.begin(), initial.end(), comp);

  int max_node_count = 0;
  int index = initial[0];
  for (int i = 0; i < initial.size(); i++) {
    bool is_same = is_same_color(initial[i], colors, initial);
    if (!is_same && color_count[colors[initial[i]]] > max_node_count) {
      max_node_count = color_count[colors[initial[i]]];
      index = initial[i];
    }
  }
  return index;
}

int main() {
  std::vector<std::vector<int>> graph = {{1,1,0}, 
                                         {1,1,0},
                                         {0,0,1}};
  std::vector<int> init = {0, 1, 2};
  std::cout << minMalwareSpread(graph, init) << std::endl;
  return 0;
}