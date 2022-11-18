#include <vector>
#include <stack>
struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

void pre_order_traserval(TreeNode* root, std::vector<int>& vec) {
  TreeNode* p = root;
  std::stack<TreeNode*> s;
  while (p != nullptr && !s.empty()) {
    while (p != nullptr) {
      vec.push_back(p->val);
      s.push(p);
      p = p->left;
    }
    p = s.top();
    p = p->right;
    s.pop();
  }
}