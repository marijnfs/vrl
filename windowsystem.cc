#include "windowsystem.h"
#include "util.h"
#include "utilvr.h"

using namespace std;

WindowSystem::WindowSystem() : width(800), height(800) {
  
}

WindowSystem::~WindowSystem() {
  if (window) {
    SDL_DestroyWindow(window);
    SDL_Quit();
    window = 0;
  }
}

void WindowSystem::init() {
	sdl_check(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER));

	init_window();
}

void WindowSystem::init_window() {
	cout << "in setup window" << endl;
	int pos_x = 700;
	int pos_y = 100;
	Uint32 wflags = SDL_WINDOW_SHOWN;

	window = SDL_CreateWindow( "hellovr [Vulkan]", pos_x, pos_y, width, height, wflags );
	if (!window) {
		cerr << "SDL Window problem: " << SDL_GetError() << endl;
		throw "";
	}

	cout << "setting title" << endl;

	string title("Window Title");
	SDL_SetWindowTitle( window, title.c_str() );
}

void WindowSystem::setup() {
	vector<Pos2Tex2> verts;

	//left eye verts
	verts.push_back( Pos2Tex2{ Vector2(-1, -1), Vector2(0, 1)});
	verts.push_back( Pos2Tex2{ Vector2(0, -1), Vector2(1, 1)} );
	verts.push_back( Pos2Tex2{ Vector2(-1, 1), Vector2(0, 0)} );
	verts.push_back( Pos2Tex2{ Vector2(0, 1), Vector2(1, 0) });

// right eye verts
	verts.push_back( Pos2Tex2{ Vector2(0, -1), Vector2(0, 1)} );
	verts.push_back( Pos2Tex2{ Vector2(1, -1), Vector2(1, 1)} );
	verts.push_back( Pos2Tex2{ Vector2(0, 1), Vector2(0, 0)} );
	verts.push_back( Pos2Tex2{ Vector2(1, 1), Vector2(1, 0)} );

	vector<uint16_t> indices = { 0, 1, 3,   0, 3, 2,   4, 5, 7,   4, 7, 6};

	cout << "setting up buffers" << endl;

//aTODO: add initialisation
	vertex_buf.init(verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, HOST);
	index_buf.init(indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, HOST);
    
    //framebuffer = new FrameRenderBuffer();
    // framebuffer->init(width, height);
	cout << " done with buffers" << endl;
}

void WindowSystem::destroy_buffers() {
  vertex_buf.destroy();
  index_buf.destroy();
}

void WindowSystem::show_message(string str) {
	SDL_ShowSimpleMessageBox( SDL_MESSAGEBOX_ERROR, "Notice", str.c_str(), NULL );
}
