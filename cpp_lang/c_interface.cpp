#include <iostream>

// C语言对外接口
// pure_c_variant.h
#ifdef __cplusplus
extern "C" {
#endif

#define VARIANT_TYPE_NULL 0
#define VARIANT_TYPE_TEXT 1
#define VARIANT_TYPE_BINARY 2

typedef struct _pure_c_variant {
  void (*set_data)(struct _pure_c_variant* self, int type, const char* data, int size);

  int (*get_data)(struct _pure_c_variant* self, char** data);

  int (*get_type)(struct _pure_c_variant* self);

  void (*close)(struct _pure_c_variant* self);
} pure_c_variant;

#ifdef __cplusplus
}
#endif

// C++实现
// C++的header, variant.h
class variant final : public pure_c_variant {
 public:
  variant();
  ~variant() = default;
  variant(const variant &other);
  variant(variant &&other) noexcept;
  variant &operator=(variant &&other) noexcept;

  static inline variant from() { return variant(); }

  /*
   * 此函数接收下述2种类型：
   * 1、std::string：有内存拷贝
   * 2、std::string_view：没有内存拷贝，引用外部内存
   */
  template <class T>
  static variant from(T data, bool binary = false) {
    variant var;
    var._hold = std::move(data);
    var._is_binary = binary;
    return var;
  }

  bool is_empty() const noexcept { return _hold.empty(); }

  bool is_text() const noexcept { return !_is_binary; }

  bool is_binary() const noexcept { return _is_binary; }

  auto view() const noexcept -> std::string;

  auto size() const noexcept -> std::size_t;

  void swap(variant &other) noexcept;

  // 脱离掌控，调用 pure_c_variant->close() 释放
  variant *detach() noexcept {
    auto detached = new variant();
    swap(*detached);
    return detached;
  }

  std::string release() {
    std::string ret = _hold;
    _hold.clear();
    return ret;
  }

 private:
  int _on_get_type() noexcept;

  int _on_get_data(char **data) noexcept;

  void _on_set_data(int type, const char *data, int size);

 public:
  std::string _hold;
  bool _is_binary = false;
};

// C++实现
// C++的实现文件, variant.cpp

variant::variant() : _is_binary(false) {
    pure_c_variant::close = [](pure_c_variant *self) noexcept {
        auto px = static_cast<variant *>(self);
        delete px;
    };

    pure_c_variant::get_data = [](pure_c_variant *self, char **data) noexcept -> int {
        auto px = static_cast<variant *>(self);
        return px->_on_get_data(data);
    };

    pure_c_variant::get_type = [](pure_c_variant *self) noexcept -> int {
        auto px = static_cast<variant *>(self);
        return px->_on_get_type();
    };

    pure_c_variant::set_data = [](pure_c_variant *self, int type, const char *data, int size) noexcept {
        auto px = static_cast<variant *>(self);
        px->_on_set_data(type, data, size);
    };
}

variant::variant(const variant &other) {
    this->_hold = other._hold;
    this->_is_binary = other._is_binary;
    this->set_data = other.set_data;
    this->get_data = other.get_data;
    this->get_type = other.get_type;
    this->close = other.close;
}

variant::variant(variant &&other) noexcept: variant() {
    swap(other);
}

variant& variant::operator=(variant &&other) noexcept {
    swap(other);
    return *this;
}

auto variant::view() const noexcept -> std::string {
    return _hold;
}

auto variant::size() const noexcept -> std::size_t {
    return _hold.size();
}

void variant::swap(variant &other) noexcept {
    if (this == &other) {
        return;
    }

    std::string temp = other._hold;
    bool temp_is_binary = other._is_binary;
    void (*set_data_)(_pure_c_variant *, int, const char *, int) =   other.set_data;
    int (*get_data_)(_pure_c_variant *, char **) =   other.get_data;
    int (*get_type_)(_pure_c_variant *) =   other.get_type;
    void (*close_)(_pure_c_variant *) =  other.close;

    other._hold = this->_hold;
    other._is_binary = this->_is_binary;
    other.get_data = this->get_data;
    other.set_data = this->set_data;
    other.get_type = this->get_type;
    other.close = this->close;

    this->_hold = temp;
    this->_is_binary = temp_is_binary;
    this->get_type = get_type_;
    this->get_data = get_data_;
    this->set_data = set_data_;
    this->close = close_;
}

int variant::_on_get_type() noexcept {
    if (is_text()) {
        return VARIANT_TYPE_TEXT;
    }

    if (is_binary()) {
        return VARIANT_TYPE_BINARY;
    }

    return VARIANT_TYPE_NULL;
}

int variant::_on_get_data(char **data) noexcept {
  auto len = _hold.size();
  if (len == 0) {
    *data = nullptr;
    return 0;
  }

  *data = const_cast<char *>(_hold.c_str());
  return static_cast<int>(len);
}

void variant::_on_set_data(int type, const char *data, int size) {
  switch (type) {
    case VARIANT_TYPE_TEXT:
      _hold = (data == nullptr || size == 0) ? "" : (size > 0) ? std::string(data, size) : std::string(data);
      _is_binary = false;
      break;
    case VARIANT_TYPE_BINARY:
      _hold = std::string(data, size);
      _is_binary = true;
      break;
    default:
      std::cout << "set data error\n";
      break;
  }
}

// 测试样例
void test1() {
  variant v1;
  v1._hold = "Hello\n";
}


int main() {
  test1();
  return 0;
}