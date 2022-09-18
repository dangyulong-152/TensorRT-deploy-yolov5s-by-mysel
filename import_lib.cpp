/*
 ����ļ���windows����Ч��linux�������
*/

#if defined(_WIN32)
#	define U_OS_WINDOWS
#else
#   define U_OS_LINUX
#endif

#ifdef U_OS_WINDOWS
#if defined(_DEBUG)
#	pragma comment(lib, "opencv_world460d.lib")
#else
#	pragma comment(lib, "opencv_world460.lib")
#endif

//����cuda
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

//����tensorRT
#pragma comment(lib, "nvinfer.lib")
#pragma comment(lib, "nvinfer_plugin.lib")
#pragma comment(lib, "nvonnxparser.lib")

#endif // U_OS_WINDOWS