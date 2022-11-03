#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> elems;
  std::string name;
  std::string value;
  int index;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == ',') {
      index = i;
    }
  }
  std::string s_copy(s);
  name = s_copy.substr(0, index);
  value = s_copy.substr(index + 1, s_copy.size() - index - 1);
  elems.push_back(name);
  elems.push_back(value);
  return elems;
}

std::string Split2(std::string name) {
  std::string ret;
  int i = 0;
  while(i < name.size()) {
    if (name[i] == '_') {
      break;
    }
    ret += name[i];
    i++;
  }
  return ret;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("usage: %s input.csv output.csv\n", argv[0]);
    return 0;
  }
  std::ifstream ifs;
  std::ofstream ofs;
  ifs.open(std::string(argv[1]));
  ofs.open(std::string(argv[2]));

  std::string line;
  double sum = 0.0;
  std::map<std::string, double> cost_map;
  while (std::getline(ifs, line)) {
    auto vec = Split(line);
    auto name = Split2(vec[0]);
    double val = stod(vec[1]);
    if(cost_map.find(name) == cost_map.end()) {
      cost_map[name] = val;
    } else {
      cost_map[name] += val;
    }
    sum += val;
  }
  std::cout << sum << std::endl;

  for (auto iter : cost_map) {
    ofs << iter.first << "," << iter.second << std::endl;
  }

  return 0;
}