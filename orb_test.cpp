#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace chrono;

int main( int argc, char** argv )
{
  if ( argc !=3 )
  {
    cout<<" 请输入两张图片的地址！ "<<endl;
    return 1;
  }
  
  cout<<" ##################  ORB特征点提取及匹配演示程序  ################## "<<endl;
  //从参数中读出图像
  Mat image1 = imread( argv[1], 6 );
  Mat image2 = imread( argv[2], 6 );
  
  //定义存储关键点的容器
  vector<KeyPoint> keypoints1,keypoints2;
  
  //定义存储描述子的容器
  Mat descriptors1,descriptors2;
  
  //定义ORB特征点对象并进行特征提取
  //CV_WRAP explicit ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31,
  //int firstLevel = 0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31 );
  ORB orb_obj( 3000, 1.2f, 0, 31, 0, 2, ORB::HARRIS_SCORE, 31 );
  steady_clock::time_point t1 = steady_clock::now();
  orb_obj.detect(image1,keypoints1);
  orb_obj.detect(image2,keypoints2);
  steady_clock::time_point t2 = steady_clock::now();
  duration<double> time_detect = duration_cast<duration<double>>(t2-t1);
  cout<<"提取两张图片的特征点共用了"<<time_detect.count()<<"s."<<endl;
  cout<<"图一共提取到"<<keypoints1.size()<<"个特征点。"<<endl;
  cout<<"图二共提取到"<<keypoints2.size()<<"个特征点。"<<endl;
  
  //绘制所提取出的特征点
  Mat image1_out;
  Mat image2_out;
  drawKeypoints(image1,keypoints1,image1_out,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  drawKeypoints(image2,keypoints2,image2_out,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  imshow("图一提取到的ORB特征点",image1_out);
  imshow("图二提取到的ORB特征点",image2_out);
  waitKey(0);
  destroyWindow("图一提取到的ORB特征点");
  destroyWindow("图二提取到的ORB特征点");
  
  //计算描述子
  steady_clock::time_point t3 = steady_clock::now();
  orb_obj.compute(image1,keypoints1,descriptors1);
  orb_obj.compute(image2,keypoints2,descriptors2);
  steady_clock::time_point t4 = steady_clock::now();
  duration<double> time_descriptor_compute = duration_cast<duration<double>>(t4-t3);
  cout<<"计算两张图片的描述子共用了"<<time_descriptor_compute.count()<<"s."<<endl;
  
  //定义描述子匹配对象
  BFMatcher orb_matcher(NORM_L2);
  
  //定义存储匹配结果的容器
  vector<DMatch> matches;
  vector<DMatch> good_matches;
  
  //特征点通过计算描述子进行匹配
  steady_clock::time_point t5 = steady_clock::now();
  orb_matcher.match(descriptors1,descriptors2,matches);
  steady_clock::time_point t6 = steady_clock::now();
  duration<double> time_match = duration_cast<duration<double>>(t6-t5);
  cout<<"两张图片的特征点匹配共用了"<<time_match.count()<<"s."<<endl;
  cout<<"共匹配到"<<matches.size()<<"对特征点。"<<endl;
  
  //绘制匹配的点对
  Mat image_matches;
  drawMatches(image1,keypoints1,image2,keypoints2,matches,image_matches);
  imshow("筛选前图一和图二的ORB匹配结果",image_matches);
  waitKey(0);
  destroyWindow("筛选前图一和图二的ORB匹配结果");
  
  //根据距离进行筛选
  double max_dist=0,min_dist=1000;
  for( int i=0;i<matches.size();i++)
  {
    double distance = matches[i].distance;
    if(distance<min_dist)
      min_dist=distance;
    if(distance>max_dist)
      max_dist=distance;
  }
  
  cout<<"最大距离："<<max_dist<<endl<<"最小距离："<<min_dist<<endl;
  
  for( int j=0;j<matches.size();j++)
  {
      if(matches[j].distance <= 100 )
      //if(matches[j].distance <= max( 4*min_dist,30.0))
      good_matches.push_back(matches[j]);
  }
  
  cout<<"距离较近的特征点共"<<good_matches.size()<<"对。"<<endl;
  
  //绘制匹配的点对
  Mat image_good_matches;
  drawMatches(image1,keypoints1,image2,keypoints2,good_matches,image_good_matches);
  imshow("根据距离筛选后图一和图二的ORB匹配结果",image_good_matches);
  waitKey(0);
  destroyWindow("根据距离筛选后图一和图二的ORB匹配结果");
  
  //根据单应性矩阵进行筛选
  vector<Point2d> image1_matches_points;
  vector<Point2d> image2_matches_points;
  for ( int i = 0 ; i < matches.size() ; i++ )
  {
    image1_matches_points.push_back( keypoints1[ matches[i].queryIdx ].pt  );
    image2_matches_points.push_back( keypoints2[ matches[i].trainIdx ].pt  );
  }
  vector<unsigned char> listpoint;
  Mat H = findHomography( image1_matches_points, image2_matches_points, CV_RANSAC, 3, listpoint );
  vector<DMatch> perfect_matches;
  for ( int i = 0 ; i < listpoint.size() ; i++ )
  {
    if ( (int)listpoint[i] )
    {
      perfect_matches.push_back(matches[i]);
    }
  }
  
  cout<<"根据单应性矩阵筛选后，匹配到"<<perfect_matches.size()<<"对特征点。"<<endl;
  Mat image_perfect_matches;
  drawMatches(image1,keypoints1,image2,keypoints2,perfect_matches,image_perfect_matches);
  imshow("根据单应性矩阵筛选后图一和图二的ORB匹配结果",image_perfect_matches);
  waitKey(0);
  destroyWindow("根据单应性矩阵筛选后图一和图二的ORB匹配结果");
  
  return 0;
}
