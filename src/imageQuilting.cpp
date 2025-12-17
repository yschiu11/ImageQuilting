#include <map>
#include <chrono>
#include <iostream>
#include "gpu_pick.h"
#include <vector>

#include <agz-utils/console.h>
#include <agz-utils/misc.h>

#include "imageQuilting.h"
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
float computeErrorSum_Lum(const Image<float> &A, int xA, int yA,
                          const Image<float> &B, int xB, int yB, int width,
                          int height, float maxSqrErr) {
  float sqrErrSum = 0;
  for (int ix = 0, ixA = xA, ixB = xB; ix < width; ++ix, ++ixA, ++ixB) {
    for (int iy = 0, iyA = yA, iyB = yB; iy < height; ++iy, ++iyA, ++iyB) {
      sqrErrSum += agz::math::sqr((A(iyA, ixA) - B(iyB, ixB)));

      if (sqrErrSum > maxSqrErr)
        return sqrErrSum;
    }
  }
  return sqrErrSum;
}

float computeMSE_Lum(const Image<float> &src, const Image<float> &dst, int srcX,
                     int srcY, int dstX, int dstY, int blockW, int blockH,
                     int overlapW, int overlapH, float minMSE,
                     float tolerance) {
  int pixelCnt = 0;

  if (dstX <= 0 && dstY <= 0)
    return 0;
  else if (dstX > 0 && dstY <= 0)
    pixelCnt = overlapW * blockH;
  else if (dstX <= 0 && dstY > 0)
    pixelCnt = overlapH * blockW;
  else
    pixelCnt = overlapW * overlapH + overlapW * (blockH - overlapH) +
               overlapH * (blockW - overlapW);

  float sseLimit = (minMSE >= std::numeric_limits<float>::max())
                       ? std::numeric_limits<float>::max()
                       : minMSE * (1.0f + tolerance) * pixelCnt;

  if (dstX > 0 && dstY <= 0) {
    const float sqrErrSum = computeErrorSum_Lum(src, srcX, srcY, dst, 0, 0,
                                                overlapW, blockH, sseLimit);

    return sqrErrSum / pixelCnt;
  }

  if (dstX <= 0 && dstY > 0) {
    const float sqrErrSum = computeErrorSum_Lum(src, srcX, srcY, dst, 0, 0,
                                                blockW, overlapH, sseLimit);

    return sqrErrSum / pixelCnt;
  }

  const float A = computeErrorSum_Lum(src, srcX, srcY, dst, 0, 0, overlapW,
                                      overlapH, sseLimit);
  if (A > sseLimit)
    return A / pixelCnt;

  const float B =
      computeErrorSum_Lum(src, srcX, srcY + overlapH, dst, 0, overlapH,
                          overlapW, blockH - overlapH, sseLimit);
  if ((A + B) > sseLimit)
    return (A + B) / pixelCnt;

  const float C = computeErrorSum_Lum(src, srcX + overlapW, srcY, dst, overlapW,
                                      0, blockW - overlapW, overlapH, sseLimit);

  return (A + B + C) / pixelCnt;
}

struct MinCostCutUnit {
  float cost = 0;
  int lastUnitOffset = 0;
};

void computeVerticalCutCost(const Image<Float3> &A, const ImageView<Float3> &B,
                            int xA, int yA, int xB, int yB, int width,
                            int height, Image<MinCostCutUnit> &result) {
  // result(y, x) is min cost cut which passes
  //      Edge(A(xA + x, yA + y), B(xB + x + 1, yB + y))
  assert(result.width() == width - 1 && result.height() == height);

  auto edgeCost = [&](int x, int y) {
    return agz::math::sqr(abs(A(yA + y, xA + x) - B(yB + y, xB + x + 1)).lum());
  };

  for (int xi = 0; xi < width - 1; ++xi)
    result(0, xi).cost = edgeCost(xi, 0);

  for (int yi = 1; yi < height; ++yi) {
    for (int xi = 0; xi < width - 1; ++xi) {
      const float costLeft = xi > 0 ? result(yi - 1, xi - 1).cost
                                    : std::numeric_limits<float>::max();
      const float costMiddle = result(yi - 1, xi).cost;
      const float costRight = xi < width - 2
                                  ? result(yi - 1, xi + 1).cost
                                  : std::numeric_limits<float>::max();

      const float ec = edgeCost(xi, yi);

      if (costLeft <= costMiddle && costLeft <= costRight)
        result(yi, xi) = {costLeft + ec, -1};
      else if (costMiddle <= costLeft && costMiddle <= costRight)
        result(yi, xi) = {costMiddle + ec, 0};
      else
        result(yi, xi) = {costRight + ec, 1};
    }
  }
}

void computeHorizontalCutCost(const Image<Float3> &A,
                              const ImageView<Float3> &B, int xA, int yA,
                              int xB, int yB, int width, int height,
                              Image<MinCostCutUnit> &result) {
  // result(y, x) is min cost cut which passes
  //      Edge(A(xA + x, yA + y), B(xB + x, yB + y + 1))
  assert(result.width() == width && result.height() == height - 1);

  auto edgeCost = [&](int x, int y) {
    return agz::math::sqr(abs(A(yA + y, xA + x) - B(yB + y + 1, xB + x)).lum());
  };

  for (int yi = 0; yi < height - 1; ++yi)
    result(yi, 0).cost = edgeCost(0, yi);

  for (int xi = 1; xi < width; ++xi) {
    for (int yi = 0; yi < height - 1; ++yi) {
      const float costNY = yi > 0 ? result(yi - 1, xi - 1).cost
                                  : std::numeric_limits<float>::max();
      const float costMY = result(yi, xi - 1).cost;
      const float costPY = yi < height - 2 ? result(yi + 1, xi - 1).cost
                                           : std::numeric_limits<float>::max();

      const float ec = edgeCost(xi, yi);

      if (costNY <= costMY && costNY <= costPY)
        result(yi, xi) = {costNY + ec, -1};
      else if (costMY <= costNY && costMY <= costPY)
        result(yi, xi) = {costMY + ec, 0};
      else
        result(yi, xi) = {costPY + ec, 1};
    }
  }
}

/**
 * ret[yi]: the cut passes
 *      Edge(A(xA + ret[yi], yA + yi), B(xB + ret[yi] + 1, yB + yi))
 */
std::vector<int> computeVerticalMinCostCut(const Image<Float3> &A,
                                           const ImageView<Float3> &B, int xA,
                                           int yA, int xB, int yB, int width,
                                           int height) {
  thread_local static Image<MinCostCutUnit> costs;
  if (costs.size() != agz::math::vec2i{width - 1, height})
    costs.initialize(height, width - 1);

  computeVerticalCutCost(A, B, xA, yA, xB, yB, width, height, costs);

  float bestCost = std::numeric_limits<float>::max();
  int bestCostXi = -1, nextXi = -1;
  for (int xi = 0; xi < width - 1; ++xi) {
    if (costs(height - 1, xi).cost < bestCost) {
      bestCost = costs(height - 1, xi).cost;
      bestCostXi = xi;
      nextXi = xi + costs(height - 1, xi).lastUnitOffset;
    }
  }

  std::vector<int> ret(height);
  ret[height - 1] = bestCostXi;

  for (int yi = height - 2; yi >= 0; --yi) {
    ret[yi] = nextXi;
    nextXi = nextXi + costs(yi, nextXi).lastUnitOffset;
  }

  return ret;
}

/**
 * ret[xi]: the cut passes
 *      Edge(A(xA + xi, yA + ret[xi]), B(xB + xi, yB + ret[xi] + 1))
 */
std::vector<int> computeHorizontalMinCostCut(const Image<Float3> &A,
                                             const ImageView<Float3> &B, int xA,
                                             int yA, int xB, int yB, int width,
                                             int height) {
  thread_local static Image<MinCostCutUnit> costs;
  if (costs.size() != agz::math::vec2i{width, height - 1})
    costs.initialize(height - 1, width);

  computeHorizontalCutCost(A, B, xA, yA, xB, yB, width, height, costs);

  float bestCost = std::numeric_limits<float>::max();
  int bestCostYi = -1, nextYi = -1;
  for (int yi = 0; yi < height - 1; ++yi) {
    if (costs(yi, width - 1).cost < bestCost) {
      bestCost = costs(yi, width - 1).cost;
      bestCostYi = yi;
      nextYi = yi + costs(yi, width - 1).lastUnitOffset;
    }
  }

  std::vector<int> ret(width);
  ret[width - 1] = bestCostYi;

  for (int xi = width - 2; xi >= 0; --xi) {
    ret[xi] = nextYi;
    nextYi = nextYi + costs(nextYi, xi).lastUnitOffset;
  }

  return ret;
}

} // namespace

ImageQuilting::ImageQuilting()
    : blockW_(3), blockH_(3), overlapW_(1), overlapH_(1), blockTolerance_(0.1f),
      useMSEBlockSelection_(true), useMinEdgeCut_(true) {}

void ImageQuilting::setParams(int blockWidth, int blockHeight) noexcept {
  setParams(blockWidth, blockHeight, std::max(1, blockWidth / 6),
            std::max(1, blockHeight / 6), 0.1f);
}

void ImageQuilting::setParams(int blockWidth, int blockHeight, int overlapWidth,
                              int overlapHeight,
                              float blockTolerance) noexcept {
  blockW_ = blockWidth;
  blockH_ = blockHeight;
  overlapW_ = overlapWidth;
  overlapH_ = overlapHeight;
  blockTolerance_ = blockTolerance;
}

void ImageQuilting::UseMSEBlockSelection(bool use) noexcept {
  useMSEBlockSelection_ = use;
}

void ImageQuilting::UseMinEdgeCut(bool use) noexcept { useMinEdgeCut_ = use; }

Image<Float3> ImageQuilting::generate(const Image<Float3> &src,
                                      int generatedWidth,
                                      int generatedHeight) const {
  // compute the padded image size
  // X * Y block
  //     -> (X * blockW - (X - 1) * overlapX)
  //      * (Y * blockH - (Y - 1) * overlapY) pixels
  //
  // (X * overlapY - (X - 1) * overlapX) >= generatedWidth
  //     -> X >= (generatedWidth - overlapX) / (overlapY - overlapX)
  //
  // (Y * blockH - (Y - 1) * overlapY) >= generatedHeight
  //     -> Y >= (generatedHeight - overlapY) / (blockH - overlapY)

  const int blockCountX = static_cast<int>(std::ceil(
      static_cast<float>(generatedWidth - overlapW_) / (blockW_ - overlapW_)));
  const int blockCountY = static_cast<int>(std::ceil(
      static_cast<float>(generatedHeight - overlapH_) / (blockH_ - overlapH_)));

  const int imageW = blockCountX * blockW_ - (blockCountX - 1) * overlapW_;
  const int imageH = blockCountY * blockH_ - (blockCountY - 1) * overlapH_;

  // create dst image

  Image<Float3> dst(imageH, imageW);
  using clock_t = std::chrono::steady_clock;
  auto t_all0 = clock_t::now();

  double t_srcLum_ms = 0.0;
  double t_pick_ms   = 0.0;
  double t_put_ms    = 0.0;

  auto to_ms = [](auto dt) {
      return std::chrono::duration<double, std::milli>(dt).count();
  };

  // fill blocks of dst image with simple raster scan order

  std::default_random_engine rng{std::random_device()()};

  agz::console::progress_bar_t pbar(blockCountY * blockCountX, 80, '=');
  pbar.display();

  for (int blockY = 0; blockY < blockCountY; ++blockY) {
    const int y = blockY * (blockH_ - overlapH_);

    for (int blockX = 0; blockX < blockCountX; ++blockX) {
      const int x = blockX * (blockW_ - overlapW_);

      ImageView<Float3> block;
      {
    	  auto t0 = clock_t::now();
    	  block = pickSourceBlock(src, dst, x, y, rng);
    	  t_pick_ms += to_ms(clock_t::now() - t0);
      }
      {
   	  auto t0 = clock_t::now();
    	  putBlockAt(block, dst, x, y);
    	  t_put_ms += to_ms(clock_t::now() - t0);
      }

      ++pbar;
    }

    pbar.display();
  }

  pbar.done();
  auto t_all1 = clock_t::now();
  double t_all_ms = to_ms(t_all1 - t_all0);

  std::cerr << "\n[PROFILE] total(ms)=" << t_all_ms
          << " srcLum(ms)=" << t_srcLum_ms
          << " pick(ms)=" << t_pick_ms
          << " put(ms)="  << t_put_ms
          << " other(ms)=" << (t_all_ms - t_srcLum_ms - t_pick_ms - t_put_ms)
          << "\n";

  return dst.subtex(0, generatedHeight, 0, generatedWidth);
}

struct CandidateBlock {
  float mse;
  int x, y;
};
ImageView<Float3> ImageQuilting::pickSourceBlock(
    const Image<Float3>         &src,
    const Image<Float3>         &dst,
    int                          x,
    int                          y,
    std::default_random_engine  &rng) const
{
    Image<float> srcLumTex = src.map([](const Float3 &c) { return c.lum(); });

    Image<float> dstLumTex(blockH_, blockW_);
    for(int iy = 0; iy < blockH_; ++iy)
        for(int ix = 0; ix < blockW_; ++ix)
            dstLumTex(iy, ix) = 0.0f;

    if (y > 0) {
        for(int iy = 0; iy < overlapH_; ++iy)
            for(int ix = 0; ix < blockW_; ++ix)
                dstLumTex(iy, ix) = dst(y + iy, x + ix).lum();
    }
    if (x > 0) {
        for(int iy = 0; iy < blockH_; ++iy)
            for(int ix = 0; ix < overlapW_; ++ix)
                dstLumTex(iy, ix) = dst(y + iy, x + ix).lum();
    }

    const int srcW = src.width();
    const int srcH = src.height();
    std::vector<float> h_srcLum((size_t)srcW * (size_t)srcH);
    for(int iy = 0; iy < srcH; ++iy)
        for(int ix = 0; ix < srcW; ++ix)
            h_srcLum[(size_t)iy * (size_t)srcW + (size_t)ix] = srcLumTex(iy, ix);

    std::vector<float> h_dstLum((size_t)blockW_ * (size_t)blockH_);
    for(int iy = 0; iy < blockH_; ++iy)
        for(int ix = 0; ix < blockW_; ++ix)
            h_dstLum[(size_t)iy * (size_t)blockW_ + (size_t)ix] = dstLumTex(iy, ix);

    unsigned int seed = (unsigned int)rng(); //  rng
    int bestX = 0, bestY = 0;
    float minMSE = 0.0f;

    bool ok = gpu_pick_block_mse(
        h_srcLum.data(), srcW, srcH,
        h_dstLum.data(), blockW_, blockH_,
        overlapW_, overlapH_,
        x, y,
        blockTolerance_,
        seed,
        bestX, bestY,
        minMSE
    );

    if(!ok)
    {
        std::uniform_int_distribution disX(0, static_cast<int>(src.width() - blockW_ - 1));
        std::uniform_int_distribution disY(0, static_cast<int>(src.height() - blockH_ - 1));
        bestX = disX(rng);
        bestY = disY(rng);
"${PROJECT_SOURCE_DIR}/src/*.h"    }

    return src.subview(bestY, bestY + blockH_, bestX, bestX + blockW_);
}

void ImageQuilting::putBlockAt(const ImageView<Float3> &block,
                               Image<Float3> &dst, int x, int y) const {
  if (x > 0 && y == 0) {
    const auto verticalCut =
        computeVerticalMinCostCut(dst, block, x, y, 0, 0, overlapW_, blockH_);

    for (int yi = 0; yi < block.height(); ++yi) {
      for (int xi = 0; xi < block.width(); ++xi) {
        if (xi > verticalCut[yi])
          dst(y + yi, x + xi) = block(yi, xi);
      }
    }
  } else if (x == 0 && y > 0) {
    const auto horizontalCut =
        computeHorizontalMinCostCut(dst, block, x, y, 0, 0, blockW_, overlapH_);

    for (int yi = 0; yi < block.height(); ++yi) {
      for (int xi = 0; xi < block.width(); ++xi) {
        if (yi > horizontalCut[xi])
          dst(y + yi, x + xi) = block(yi, xi);
      }
    }
  } else if (x > 0 && y > 0) {
    const auto verticalCut =
        computeVerticalMinCostCut(dst, block, x, y, 0, 0, overlapW_, blockH_);

    const auto horizontalCut =
        computeHorizontalMinCostCut(dst, block, x, y, 0, 0, blockW_, overlapH_);

    for (int yi = 0; yi < block.height(); ++yi) {
      for (int xi = 0; xi < block.width(); ++xi) {
        if (xi > verticalCut[yi] && yi > horizontalCut[xi])
          dst(y + yi, x + xi) = block(yi, xi);
      }
    }
  } else {
    for (int yi = 0; yi < block.height(); ++yi) {
      for (int xi = 0; xi < block.width(); ++xi)
        dst(y + yi, x + xi) = block(yi, xi);
    }
  }
}
