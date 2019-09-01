#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <cmath>
#include <sstream>
#include <chrono>
#include <thread>

using namespace std;

#include "scene.h"
#include "db.h"
#include "serialise.h"



int main() {
  Scene scene;

  
  cout << "start" << endl;

  Pos p1{0, 0, 1};
  Pos p2{0, 0, 1};
  Pos p3{0, 0, 1};

  
  
  scene.add_screen("o1");
  scene.add_screen("o2");
  scene.add_screen("o3");

  scene.set_pos("o1", p1);
  scene.set_pos("o2", p2);
  scene.set_pos("o3", p3);

  int goal = rand() % 3;

  
  
  while (scene.time < 2000) {
    
    scene.step();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000/60));
      
  }

  cout << "end" << endl;  
    
}
