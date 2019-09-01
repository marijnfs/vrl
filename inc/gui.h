#pragma once

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_SHADER_VERSION "#version 300 es\n"

#define MAX_VERTEX_MEMORY 512 * 1024
#define MAX_ELEMENT_MEMORY 128 * 1024

#include "nuklear.h"

enum { EASY, HARD };


struct GUI {
	nk_context ctx;
	nk_font_atlas atlas;
	//nk_user_font font;
	nk_font *font;

	int op = EASY;
	float value = 0.6f;
	int i = 20;
	int w, h;

	GUI() {
        nk_font_atlas_init_default(&atlas);
        nk_font_atlas_begin(&atlas);
        font = nk_font_atlas_add_default(&atlas, 13, 0);
        const void* img = nk_font_atlas_bake(&atlas, &w, &h, NK_FONT_ATLAS_RGBA32);
        //nk_font_atlas_end(&atlas, nk_handle_id(texture), 0);
        nk_font_atlas_end(&atlas, nk_handle_id(0), 0);
        nk_init_default(&ctx, &font->handle);

		if (nk_begin(&ctx, "Show", nk_rect(50, 50, 220, 220),
			NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_CLOSABLE)) {
			/* fixed widget pixel width */
			nk_layout_row_static(&ctx, 30, 80, 1);
			if (nk_button_label(&ctx, "button")) {
				/* event handling */
			}

			/* fixed widget window ratio width */
			nk_layout_row_dynamic(&ctx, 30, 2);
			if (nk_option_label(&ctx, "easy", op == EASY)) op = EASY;
			if (nk_option_label(&ctx, "hard", op == HARD)) op = HARD;

			/* custom widget pixel width */
			nk_layout_row_begin(&ctx, NK_STATIC, 30, 2);
			{
				nk_layout_row_push(&ctx, 50);
				nk_label(&ctx, "Volume:", NK_TEXT_LEFT);
				nk_layout_row_push(&ctx, 110);
				nk_slider_float(&ctx, 0, &value, 1.0f, 0.1f);
			}
			nk_layout_row_end(&ctx);
		}

	}

	~GUI() {
		nk_end(&ctx);
	}
};