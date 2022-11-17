#include <vector>

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

void travel(TreeNode* root, std::vector<int>& vec) {
  if (root == nullptr) {
    return;
  }
  travel(root->left, vec);
  vec.push_back(root->val);
  travel(root->right, vec);
  return;
}

std::vector<int> inorderTraversal(TreeNode *root) {
  std::vector<int> ret;
  travel(root, ret);
  return ret;
}
