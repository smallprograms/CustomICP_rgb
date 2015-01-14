#include <iostream>
#include <sstream>
#include <cmath>
#include "CustomICP.h"
#include "GraphOptimizer_G2O.h"
#include "Surf.h"
#include "BashUtils.h"
#include "opencv2/core/core.hpp"
#include "utils.h"
#include "AlignSystem.h"


int main (int argc, char** argv)
{

    //path of files to load, min and max index of files
    int min;
    int max;
    char* path;
    char* outFile;
    char* global; //previously constructed cloud path
    char* groundTruth; //ground truth cam movement path

    bool loadedGlobal=false;

    if( argc < 5) {

        std::cout << "Usage:\n " << argv[0] << " [global_cloud] path min_index max_index out_file\n";
        std::cout << "Example:\n " << argv[0] << " car 2 10 car.pcd\n";
        std::cout << "Will use car/cap2 until car/cap10 and save the aligned clouds in car.pcd\n";
        std::cout <<  argv[0] << " global_car.pcd car 2 10 car.pcd\n";
        std::cout << "Will load and apply global_car.pcd.transf and then add  car/cap2 until car/cap10 to global_car.pcd and save the aligned clouds in car.pcd\n";
        std::cout <<  argv[0] << " movements.txt car 2 10 car.pcd\n";
        std::cout << "Will apply movements to each cloud and will generate a global cloud car.pcd with them\n";
        std::cout <<  argv[0] << " car.pcd car.pcd.optimized_poses\n";
        std::cout << "Will run graph optimizer (it will read car.pcd.movements,car.pcd..from_to.txt,car.pcd.fitness.txt,car.pcd.relativeTransf.txt to read graph) \n";
        return 0;


    } else {

        if(argc >= 6) {

            min = atoi(argv[3]);
            max = atoi(argv[4]);
            path = argv[2];
            outFile = argv[5];

            int increment = 1;
            if( argc == 7 ) {
                increment = atoi(argv[6]);
            }

            if( endswith(argv[1], ".pcd") == false ) {

                groundTruth = argv[1];
                AlignSystem alignSys(path, outFile, global, min, max);
                alignSys.groundTruthAlign( groundTruth, increment );
                return 0;

            } else {
                global = argv[1];
                loadedGlobal = true;
            }



        } else {

            min = atoi(argv[2]);
            max = atoi(argv[3]);
            path = argv[1];
            outFile = argv[4];

        }

    }

    AlignSystem alignSys( path, outFile, global, min, max );
    alignSys.align(loadedGlobal);
}
