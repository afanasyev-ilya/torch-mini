#include <gtest/gtest.h>
#include <eltwise.hpp>
#include <vector>

TEST(EltwiseAdd, HappyPath) {
  std::vector<float> a{1.f, 2.f, 3.f};
  std::vector<float> b{4.f, 5.f, 6.f};

  auto c = torch_mini::eltwise_add(a, b);
  ASSERT_EQ(c.size(), 3u);
  EXPECT_FLOAT_EQ(c[0], 5.f);
  EXPECT_FLOAT_EQ(c[1], 7.f);
  EXPECT_FLOAT_EQ(c[2], 9.f);
}

TEST(EltwiseAdd, SizeMismatch) {
  std::vector<float> a{1.f, 2.f};
  std::vector<float> b{3.f};
  EXPECT_THROW({ auto _ = torch_mini::eltwise_add(a, b); }, std::invalid_argument);
}
