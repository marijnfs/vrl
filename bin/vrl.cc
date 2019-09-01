#include "physics.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <openvr.h>

#include "convert.h"
#include "database.h"
#include "frame.capnp.h"

#include "lodepng.h"
#include "utils.h"

#include "gui.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include "SimpleOpenGL3App.h"
#include "OpenGLInclude.h"

using namespace std;


void ThreadSleep(unsigned long nMilliseconds)
{
#if defined(_WIN32)
	::Sleep(nMilliseconds);
#elif defined(POSIX)
	usleep(nMilliseconds * 1000);
#endif
}

struct VertexDataScene
{
	Vector3 position;
	Vector2 texCoord;
};

struct VertexDataLens
{
	Vector2 position;
	Vector2 texCoordRed;
	Vector2 texCoordGreen;
	Vector2 texCoordBlue;
};

struct FramebufferDesc
{
	GLuint m_nDepthBufferId;
	GLuint m_nRenderTextureId;
	GLuint m_nRenderFramebufferId;
	GLuint m_nResolveTextureId;
	GLuint m_nResolveFramebufferId;
};

inline void check(vr::EVRInitError &error) {
	if (error != vr::VRInitError_None)
	{
		char buf[1024];
		//sprintf_s(buf, sizeof(buf), "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(error));
		std::ostringstream oss;
		oss << "Unable to init VR runtime: " << vr::VR_GetVRInitErrorAsEnglishDescription(error);
		cerr << "VR_Init Failed" << buf << endl;
		vr::VR_Shutdown();

		throw false;
	}
}

inline Matrix4 convert_stream_vrmatrix_to_matrix4( const vr::HmdMatrix34_t &matPose )
{
	Matrix4 matrixObj(
		matPose.m[0][0], matPose.m[1][0], matPose.m[2][0], 0.0,
		matPose.m[0][1], matPose.m[1][1], matPose.m[2][1], 0.0,
		matPose.m[0][2], matPose.m[1][2], matPose.m[2][2], 0.0,
		matPose.m[0][3], matPose.m[1][3], matPose.m[2][3], 1.0f
		);
	return matrixObj;
}

inline GLuint CompileGLShader( const char *pchShaderName, const char *pchVertexShader, const char *pchFragmentShader )
{
	GLuint unProgramID = glCreateProgram();

	GLuint nSceneVertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource( nSceneVertexShader, 1, &pchVertexShader, NULL);
	glCompileShader( nSceneVertexShader );

	GLint vShaderCompiled = GL_FALSE;
	glGetShaderiv( nSceneVertexShader, GL_COMPILE_STATUS, &vShaderCompiled);
	if ( vShaderCompiled != GL_TRUE)
	{
		b3Printf("%s - Unable to compile vertex shader %d!\n", pchShaderName, nSceneVertexShader);
		glDeleteProgram( unProgramID );
		glDeleteShader( nSceneVertexShader );
		return 0;
	}
	glAttachShader( unProgramID, nSceneVertexShader);
	glDeleteShader( nSceneVertexShader ); // the program hangs onto this once it's attached

	GLuint  nSceneFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource( nSceneFragmentShader, 1, &pchFragmentShader, NULL);
	glCompileShader( nSceneFragmentShader );

	GLint fShaderCompiled = GL_FALSE;
	glGetShaderiv( nSceneFragmentShader, GL_COMPILE_STATUS, &fShaderCompiled);
	if (fShaderCompiled != GL_TRUE)
	{
		b3Printf("%s - Unable to compile fragment shader %d!\n", pchShaderName, nSceneFragmentShader );
		glDeleteProgram( unProgramID );
		glDeleteShader( nSceneFragmentShader );
		return 0;	
	}

	glAttachShader( unProgramID, nSceneFragmentShader );
	glDeleteShader( nSceneFragmentShader ); // the program hangs onto this once it's attached

	glLinkProgram( unProgramID );

	GLint programSuccess = GL_TRUE;
	glGetProgramiv( unProgramID, GL_LINK_STATUS, &programSuccess);
	if ( programSuccess != GL_TRUE )
	{
		b3Printf("%s - Error linking program %d!\n", pchShaderName, unProgramID);
		glDeleteProgram( unProgramID );
		return 0;
	}

	glUseProgram( unProgramID );
	glUseProgram( 0 );

	return unProgramID;
}

void add_cube_vertex( float fl0, float fl1, float fl2, float fl3, float fl4, std::vector<float> &vertdata )
{
	vertdata.push_back( fl0 );
	vertdata.push_back( fl1 );
	vertdata.push_back( fl2 );
	vertdata.push_back( fl3 );
	vertdata.push_back( fl4 );
}


//-----------------------------------------------------------------------------
// Purpose:
//-----------------------------------------------------------------------------
void add_cube_to_scene( Matrix4 mat, std::vector<float> &vertdata, float s = 1.0 )
{
	// Matrix4 mat( outermat.data() );

	Vector4 A = mat * Vector4( 0, 0, 0, 1 );
	Vector4 B = mat * Vector4( s, 0, 0, 1 );
	Vector4 C = mat * Vector4( s, s, 0, 1 );
	Vector4 D = mat * Vector4( 0, s, 0, 1 );
	Vector4 E = mat * Vector4( 0, 0, s, 1 );
	Vector4 F = mat * Vector4( s, 0, s, 1 );
	Vector4 G = mat * Vector4( s, s, s, 1 );
	Vector4 H = mat * Vector4( 0, s, s, 1 );

	// triangles instead of quads
	add_cube_vertex( E.x, E.y, E.z, 0, 1, vertdata ); //Front
	add_cube_vertex( F.x, F.y, F.z, 1, 1, vertdata );
	add_cube_vertex( G.x, G.y, G.z, 1, 0, vertdata );
	add_cube_vertex( G.x, G.y, G.z, 1, 0, vertdata );
	add_cube_vertex( H.x, H.y, H.z, 0, 0, vertdata );
	add_cube_vertex( E.x, E.y, E.z, 0, 1, vertdata );
					 
	add_cube_vertex( B.x, B.y, B.z, 0, 1, vertdata ); //Back
	add_cube_vertex( A.x, A.y, A.z, 1, 1, vertdata );
	add_cube_vertex( D.x, D.y, D.z, 1, 0, vertdata );
	add_cube_vertex( D.x, D.y, D.z, 1, 0, vertdata );
	add_cube_vertex( C.x, C.y, C.z, 0, 0, vertdata );
	add_cube_vertex( B.x, B.y, B.z, 0, 1, vertdata );
					
	add_cube_vertex( H.x, H.y, H.z, 0, 1, vertdata ); //Top
	add_cube_vertex( G.x, G.y, G.z, 1, 1, vertdata );
	add_cube_vertex( C.x, C.y, C.z, 1, 0, vertdata );
	add_cube_vertex( C.x, C.y, C.z, 1, 0, vertdata );
	add_cube_vertex( D.x, D.y, D.z, 0, 0, vertdata );
	add_cube_vertex( H.x, H.y, H.z, 0, 1, vertdata );
				
	add_cube_vertex( A.x, A.y, A.z, 0, 1, vertdata ); //Bottom
	add_cube_vertex( B.x, B.y, B.z, 1, 1, vertdata );
	add_cube_vertex( F.x, F.y, F.z, 1, 0, vertdata );
	add_cube_vertex( F.x, F.y, F.z, 1, 0, vertdata );
	add_cube_vertex( E.x, E.y, E.z, 0, 0, vertdata );
	add_cube_vertex( A.x, A.y, A.z, 0, 1, vertdata );
					
	add_cube_vertex( A.x, A.y, A.z, 0, 1, vertdata ); //Left
	add_cube_vertex( E.x, E.y, E.z, 1, 1, vertdata );
	add_cube_vertex( H.x, H.y, H.z, 1, 0, vertdata );
	add_cube_vertex( H.x, H.y, H.z, 1, 0, vertdata );
	add_cube_vertex( D.x, D.y, D.z, 0, 0, vertdata );
	add_cube_vertex( A.x, A.y, A.z, 0, 1, vertdata );

	add_cube_vertex( F.x, F.y, F.z, 0, 1, vertdata ); //Right
	add_cube_vertex( B.x, B.y, B.z, 1, 1, vertdata );
	add_cube_vertex( C.x, C.y, C.z, 1, 0, vertdata );
	add_cube_vertex( C.x, C.y, C.z, 1, 0, vertdata );
	add_cube_vertex( G.x, G.y, G.z, 0, 0, vertdata );
	add_cube_vertex( F.x, F.y, F.z, 0, 1, vertdata );
}

typedef void (*vr_controller_input_callback)(int controllerId, int button, int state, float pos[4], float orientation[4]);
typedef void (*vr_controller_move_callback)(int controllerId, float pos[4], float orientation[4], float analogAxis);

inline std::string get_tracked_device_string(vr::IVRSystem *pHmd, vr::TrackedDeviceIndex_t unDevice, vr::TrackedDeviceProperty prop, vr::TrackedPropertyError *error = NULL) {
	uint32_t unRequiredBufferLen = pHmd->GetStringTrackedDeviceProperty(unDevice, prop, NULL, 0, error);
	if (unRequiredBufferLen == 0)
		return "";

	char *pchBuffer = new char[unRequiredBufferLen];
	unRequiredBufferLen = pHmd->GetStringTrackedDeviceProperty(unDevice, prop, pchBuffer, unRequiredBufferLen, error);
	std::string result = pchBuffer;
	delete[] pchBuffer;
	return result;
}

void vr_controller_input_callback_printer(int controllerId, int button, int state, float pos[4], float orientation[4]) {
	cout << "id: " << controllerId << " b: " << button << " s: " << state << " pos: ";
	for (int i(0); i < 4; ++i) cout << pos[i] << ",";
	cout << " -- or: ";
	for (int i(0); i < 4; ++i) cout << orientation[i] << ",";
	cout << endl;
}

void vr_controller_move_callback_printer(int controllerId, float pos[4], float orientation[4], float analogAxis) {
	cout << "id: " << controllerId << " pos: ";
	for (int i(0); i < 4; ++i) cout << pos[i] << ",";
	cout << " -- or: ";
	for (int i(0); i < 4; ++i) cout << orientation[i] << ",";
	cout << " analog: " << analogAxis << endl;
}


struct VRSetup {
	VRSetup()
	{
		vr::EVRInitError error(vr::VRInitError_None);
		hmd = vr::VR_Init(&error, vr::VRApplication_Scene);
		check(error);

		render_models = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &error);
		check(error);

		window_width = 1280;
		window_height = 720;

		near_clip = .1f;
		far_clip = 3000.0f;

		fscale = .3f;
		fscale_spacing = 4.0f;

		scene_volume_dim = 20;

		driver_str = get_tracked_device_string(hmd, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_TrackingSystemName_String);
		display_str = get_tracked_device_string(hmd, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SerialNumber_String);


		if (!vr::VRCompositor())
		{
			cerr << "Compositor initialization failed. See log file for details\n" << endl;
			throw false;
		}

		initGL();
		init();
	}

	~VRSetup() {
		vr::VR_Shutdown();
	}

	void initGL() {
		app = new SimpleOpenGL3App("SimpleOpenGL3App", window_width, window_height, true);

		if (!create_all_shaders())
			throw StringException("Shhaaait shaders");

		glewExperimental = GL_TRUE;
		glewInit();
		setup_cameras();
	}

	void init() {
		// if( !CreateAllShaders() )
		// 	return false;

		// SetupTexturemaps();
		// SetupScene();
		// SetupCameras();
		// SetupStereoRenderTargets();
		// SetupDistortion();

		// SetupRenderModels();

	  cout << "Setting up: textures, scene cameras, stereo targets, distorion" << endl;
		setup_texture_maps();
		setup_scene();
		setup_cameras();
		setup_stereo_render_targets();
		setup_distortion();
		//setup_render_models();
		cout << "Done setting up: textures, scene cameras, stereo targets, distorion" << endl;

	}

	void get_controller_transform(int unDevice, b3Transform& tr) {
		const Matrix4 & matOrg = tracked_device_pose_mat[unDevice];
		tr.setIdentity();
		tr.setOrigin(b3MakeVector3(matOrg[12], matOrg[13], matOrg[14]));//pos[1]));
		b3Matrix3x3 bmat;
		for (int i = 0; i<3; i++)
		{
			for (int j = 0; j<3; j++)
			{
				bmat[i][j] = matOrg[i + 4 * j];
			}
		}
		tr.setBasis(bmat);
		b3Transform y2z;
		y2z.setIdentity();
		y2z.setRotation(b3Quaternion(0, B3_HALF_PI, 0));
		tr = y2z*tr;
	}

	bool handle_vr_input(vr_controller_input_callback input_callback = 0, vr_controller_move_callback move_callback = 0) {
		bool ret = false;

		// Process SteamVR events
		vr::VREvent_t event;
		while (hmd->PollNextEvent(&event, sizeof(event)))
		{
			process_vr_event(event);
		}

		// Process SteamVR controller state
		for (vr::TrackedDeviceIndex_t unDevice = 0; unDevice < vr::k_unMaxTrackedDeviceCount; unDevice++)
		{
			vr::VRControllerState_t state;
			if (hmd->GetControllerState(unDevice, &state, sizeof(vr::VRControllerState_t)))
			{
				//we need to have the 'move' events, so no early out here
				//if (sPrevStates[unDevice].unPacketNum != state.unPacketNum)
				if (hmd->GetTrackedDeviceClass(unDevice) == vr::TrackedDeviceClass_Controller)
				{
					prev_controller_states[unDevice].unPacketNum = state.unPacketNum;

					for (int button = 0; button < vr::k_EButton_Max; button++)
					{
						uint64_t trigger = vr::ButtonMaskFromId((vr::EVRButtonId)button);

						bool isTrigger = (state.ulButtonPressed&trigger) != 0;
						if (isTrigger)
						{

							b3Transform tr;
							get_controller_transform(unDevice, tr);
							float pos[3] = { tr.getOrigin()[0], tr.getOrigin()[1], tr.getOrigin()[2] };
							b3Quaternion born = tr.getRotation();
							float orn[4] = { born[0], born[1], born[2], born[3] };

							//pressed now, not pressed before -> raise a button down event
							if ((prev_controller_states[unDevice].ulButtonPressed&trigger) == 0)
							{
								printf("Device PRESSED: %d, button %d\n", unDevice, button);
								if (input_callback)
									(*input_callback)(unDevice, button, 1, pos, orn);
							}
							else
							{
								//							printf("Device MOVED: %d\n", unDevice);
								if (move_callback)
									(*move_callback)(unDevice, pos, orn, state.rAxis[1].x);
							}
						}
						else
						{
							if (hmd->GetTrackedDeviceClass(unDevice) == vr::TrackedDeviceClass_Controller)
							{
								b3Transform tr;
								get_controller_transform(unDevice, tr);
								float pos[3] = { tr.getOrigin()[0], tr.getOrigin()[1], tr.getOrigin()[2] };
								b3Quaternion born = tr.getRotation();
								float orn[4] = { born[0], born[1], born[2], born[3] };
								//							printf("Device RELEASED: %d, button %d\n", unDevice,button);

								//not pressed now, but pressed before -> raise a button up event
								if ((prev_controller_states[unDevice].ulButtonPressed&trigger) != 0)
								{
									if (input_callback)
										(*input_callback)(unDevice, button, 0, pos, orn);
								}
								else
								{
									if (move_callback)
										(*move_callback)(unDevice, pos, orn, state.rAxis[1].x);
								}
							}
						}
					}
				}

				//			m_rbShowTrackedDevice[ unDevice ] = state.ulButtonPressed == 0;
			}
			prev_controller_states[unDevice] = state;
		}

		return ret;
	}
	
	void process_vr_event(const vr::VREvent_t & event) {
		switch (event.eventType)
		{
			case vr::VREvent_TrackedDeviceActivated:
			{
				//setup_render_model_for_tracked_device(event.trackedDeviceIndex);
				b3Printf("Device %u attached. Setting up render model.\n", event.trackedDeviceIndex);
			} break;

			case vr::VREvent_TrackedDeviceDeactivated:
			{
				b3Printf("Device %u detached.\n", event.trackedDeviceIndex);
			} break;

			case vr::VREvent_TrackedDeviceUpdated:
			{
				b3Printf("Device %u updated.\n", event.trackedDeviceIndex);
			} break;
		}

	}

	Matrix4 get_hmd_matrix_projection_eye(vr::Hmd_Eye nEye)
	{
		if (!hmd)
			return Matrix4();

		vr::HmdMatrix44_t mat = hmd->GetProjectionMatrix(nEye, near_clip, far_clip);

		return Matrix4(
			mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
			mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
			mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
			mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
		);
	}

	Matrix4 get_hmd_matrix_pose_eye(vr::Hmd_Eye nEye)
	{
		if (!hmd)
			return Matrix4();

		vr::HmdMatrix34_t mat_eye = hmd->GetEyeToHeadTransform(nEye);
		Matrix4 matrixObj(
			mat_eye.m[0][0], mat_eye.m[1][0], mat_eye.m[2][0], 0.0,
			mat_eye.m[0][1], mat_eye.m[1][1], mat_eye.m[2][1], 0.0,
			mat_eye.m[0][2], mat_eye.m[1][2], mat_eye.m[2][2], 0.0,
			mat_eye.m[0][3], mat_eye.m[1][3], mat_eye.m[2][3], 1.0f
		);

		return matrixObj.invert();
	}


	bool setup_texture_maps()
	{
		std::string str_full_path("./data/cube_texture.png");
		
		std::vector<unsigned char> imageRGBA;
		unsigned int nImageWidth(0);
		unsigned int nImageHeight(0);
		unsigned nError = lodepng::decode( imageRGBA, nImageWidth, nImageHeight, str_full_path.c_str() );
		
		if ( nError != 0 ) {
			cout << ("Couldnt open file " + str_full_path) << endl;
			throw StringException("Couldnt open file " + str_full_path);
			exit(1);
		}


		glGenTextures(1, &cube_texture );
		glBindTexture( GL_TEXTURE_2D, cube_texture );

		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, nImageWidth, nImageHeight,
			0, GL_RGBA, GL_UNSIGNED_BYTE, &imageRGBA[0] );

		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );

		GLfloat fLargest(0);
		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest);
		 	
		glBindTexture( GL_TEXTURE_2D, 0 );

		return ( cube_texture != 0 );
	}

	void setup_cube() {
		vector<float> vertdataarray;

		Matrix4 basic_mat;
		add_cube_to_scene( basic_mat, vertdataarray, .03 );
		cube_vert_count = vertdataarray.size()/5;

		glGenVertexArrays(1, &cube_vao);
		glBindVertexArray(cube_vao);

		glGenBuffers( 1, &cube_vert_buffer );
		glBindBuffer( GL_ARRAY_BUFFER, cube_vert_buffer );
		glBufferData( GL_ARRAY_BUFFER, sizeof(float) * vertdataarray.size(), &vertdataarray[0], GL_STATIC_DRAW);

		glBindBuffer( GL_ARRAY_BUFFER, cube_vert_buffer );

		GLsizei stride = sizeof(VertexDataScene);
		uintptr_t offset = 0;

		glEnableVertexAttribArray( 0 );
		glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, stride , (const void *)offset);

		offset += sizeof(Vector3);
		glEnableVertexAttribArray( 1 );
		glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, stride, (const void *)offset);

		glBindVertexArray( 0 );
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}

	void setup_scene()
	{
		if ( !hmd )
			return;

		vector<float> vertdataarray;

		Matrix4 matScale;
		matScale.scale( fscale, fscale, fscale );
		Matrix4 matTransform;
		matTransform.translate(
			-( scene_volume_dim * fscale_spacing ) / 2.f,
			-( scene_volume_dim * fscale_spacing ) / 2.f,
			-( scene_volume_dim * fscale_spacing ) / 2.f);
		
		Matrix4 mat = matScale * matTransform;

		int volume_dim(scene_volume_dim + .5); //rough rounding
		
		for (int z = 0; z < volume_dim; z++)
		{
			for (int y = 0; y < volume_dim; y++)
			{
				for (int x = 0; x < volume_dim; x++)
				{
					add_cube_to_scene( mat, vertdataarray );
					mat = mat * Matrix4().translate( fscale_spacing, 0, 0 );
				}
				mat = mat * Matrix4().translate( -(scene_volume_dim) * fscale_spacing, fscale_spacing, 0 );
			}
			mat = mat * Matrix4().translate( 0, -(scene_volume_dim) * fscale_spacing, fscale_spacing );
		}
		scene_vert_count = vertdataarray.size()/5;
		


		glGenVertexArrays(1, &scene_vao);
		glBindVertexArray(scene_vao);

		glGenBuffers( 1, &scene_vert_buffer );
		glBindBuffer( GL_ARRAY_BUFFER, scene_vert_buffer );
		glBufferData( GL_ARRAY_BUFFER, sizeof(float) * vertdataarray.size(), &vertdataarray[0], GL_STATIC_DRAW);

		glBindBuffer( GL_ARRAY_BUFFER, scene_vert_buffer );

		GLsizei stride = sizeof(VertexDataScene);
		uintptr_t offset = 0;

		glEnableVertexAttribArray( 0 );
		glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, stride , (const void *)offset);

		offset += sizeof(Vector3);
		glEnableVertexAttribArray( 1 );
		glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, stride, (const void *)offset);

		glBindVertexArray( 0 );
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
	}

	void update_hmd_matrix_pose()
	{
		if (!hmd)
			return;
		{
			B3_PROFILE("WaitGetPoses");
			vr::VRCompositor()->WaitGetPoses(tracked_device_pose, vr::k_unMaxTrackedDeviceCount, NULL, 0);
		}

		valid_pose_count = 0;
		str_pose_classes = "";
		{
			B3_PROFILE("for loop");
			for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice)
			{
				if (tracked_device_pose[nDevice].bPoseIsValid)
				{
					valid_pose_count++;
					
					tracked_device_pose_mat[nDevice] = convert_stream_vrmatrix_to_matrix4(tracked_device_pose[nDevice].mDeviceToAbsoluteTracking);
					//cout << "device: " << nDevice << " " << tracked_device_pose_mat[nDevice] << endl;
					if (dev_class_char[nDevice] == 0)
					{
						switch (hmd->GetTrackedDeviceClass(nDevice))
						{
						case vr::TrackedDeviceClass_Controller:        dev_class_char[nDevice] = 'C'; break;
						case vr::TrackedDeviceClass_HMD:               dev_class_char[nDevice] = 'H'; break;
						case vr::TrackedDeviceClass_Invalid:           dev_class_char[nDevice] = 'I'; break;
					//case vr::TrackedDeviceClass_Other:             dev_class_char[nDevice] = 'O'; break;
						case vr::TrackedDeviceClass_TrackingReference: dev_class_char[nDevice] = 'T'; break;
						default:                                       dev_class_char[nDevice] = '?'; break;
						}
					}
					str_pose_classes += dev_class_char[nDevice];
				}
			}
		}
		{
			B3_PROFILE("mat4_hmd_pose invert");

			if (tracked_device_pose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid)
			{
				mat4_hmd_pose = tracked_device_pose_mat[vr::k_unTrackedDeviceIndex_Hmd].invert();
				//cout << "reading hdm pose: " << mat4_hmd_pose << endl;
			}
		}
	}

	void setup_distortion()
	{
		if ( !hmd )
			return;

		GLushort m_iLensGridSegmentCountH = 43;
		GLushort m_iLensGridSegmentCountV = 43;

		float w = (float)( 1.0/float(m_iLensGridSegmentCountH-1));
		float h = (float)( 1.0/float(m_iLensGridSegmentCountV-1));

		float u, v = 0;

		std::vector<VertexDataLens> vertices(0);
		VertexDataLens vert;

		//left eye distortion verts
		float Xoffset = -1;
		for( int y=0; y<m_iLensGridSegmentCountV; y++ )
		{
			for( int x=0; x<m_iLensGridSegmentCountH; x++ )
			{
				u = x*w; v = 1-y*h;
				vert.position = Vector2( Xoffset+u, -1+2*y*h );

				vr::DistortionCoordinates_t dc0;
				bool result = hmd->ComputeDistortion(vr::Eye_Left, u, v,&dc0);
				btAssert(result);
				vert.texCoordRed = Vector2(dc0.rfRed[0], 1 - dc0.rfRed[1]);
				vert.texCoordGreen =  Vector2(dc0.rfGreen[0], 1 - dc0.rfGreen[1]);
				vert.texCoordBlue = Vector2(dc0.rfBlue[0], 1 - dc0.rfBlue[1]);

				vertices.push_back( vert );
			}
		}

		//right eye distortion verts
		Xoffset = 0;
		for( int y=0; y<m_iLensGridSegmentCountV; y++ )
		{
			for( int x=0; x<m_iLensGridSegmentCountH; x++ )
			{
				u = x*w; v = 1-y*h;
				vert.position = Vector2( Xoffset+u, -1+2*y*h );

				vr::DistortionCoordinates_t dc0;
				bool result = hmd->ComputeDistortion( vr::Eye_Right, u, v,&dc0 );
				btAssert(result);
				vert.texCoordRed = Vector2(dc0.rfRed[0], 1 - dc0.rfRed[1]);
				vert.texCoordGreen = Vector2(dc0.rfGreen[0], 1 - dc0.rfGreen[1]);
				vert.texCoordBlue = Vector2(dc0.rfBlue[0], 1 - dc0.rfBlue[1]);

				vertices.push_back( vert );
			}
		}

		std::vector<GLushort> vIndices;
		GLushort a,b,c,d;

		GLushort offset = 0;
		for( GLushort y=0; y<m_iLensGridSegmentCountV-1; y++ )
		{
			for( GLushort x=0; x<m_iLensGridSegmentCountH-1; x++ )
			{
				a = m_iLensGridSegmentCountH*y+x +offset;
				b = m_iLensGridSegmentCountH*y+x+1 +offset;
				c = (y+1)*m_iLensGridSegmentCountH+x+1 +offset;
				d = (y+1)*m_iLensGridSegmentCountH+x +offset;
				vIndices.push_back( a );
				vIndices.push_back( b );
				vIndices.push_back( c );

				vIndices.push_back( a );
				vIndices.push_back( c );
				vIndices.push_back( d );
			}
		}

		offset = (m_iLensGridSegmentCountH)*(m_iLensGridSegmentCountV);
		for( GLushort y=0; y<m_iLensGridSegmentCountV-1; y++ )
		{
			for( GLushort x=0; x<m_iLensGridSegmentCountH-1; x++ )
			{
				a = m_iLensGridSegmentCountH*y+x +offset;
				b = m_iLensGridSegmentCountH*y+x+1 +offset;
				c = (y+1)*m_iLensGridSegmentCountH+x+1 +offset;
				d = (y+1)*m_iLensGridSegmentCountH+x +offset;
				vIndices.push_back( a );
				vIndices.push_back( b );
				vIndices.push_back( c );

				vIndices.push_back( a );
				vIndices.push_back( c );
				vIndices.push_back( d );
			}
		}

		ui_index_size = vIndices.size();

		glGenVertexArrays( 1, &lens_vertex_array );
		glBindVertexArray( lens_vertex_array );

		glGenBuffers( 1, &gl_id_vertex_buffer );
		glBindBuffer( GL_ARRAY_BUFFER, gl_id_vertex_buffer );
		glBufferData( GL_ARRAY_BUFFER, vertices.size()*sizeof(VertexDataLens), &vertices[0], GL_STATIC_DRAW );

		glGenBuffers( 1, &gl_id_index_buffer );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, gl_id_index_buffer );
		glBufferData( GL_ELEMENT_ARRAY_BUFFER, vIndices.size()*sizeof(GLushort), &vIndices[0], GL_STATIC_DRAW );

		glEnableVertexAttribArray( 0 );
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataLens), (void *)offsetof( VertexDataLens, position ) );

		glEnableVertexAttribArray( 1 );
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataLens), (void *)offsetof( VertexDataLens, texCoordRed ) );

		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataLens), (void *)offsetof( VertexDataLens, texCoordGreen ) );

		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(VertexDataLens), (void *)offsetof( VertexDataLens, texCoordBlue ) );

		glBindVertexArray( 0 );

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		glDisableVertexAttribArray(3);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	// CGLRenderModel *find_or_load_render_model(const char *pchRenderModelName)
	// {
	// 	CGLRenderModel *pRenderModel = NULL;
	// 	for (std::vector< CGLRenderModel * >::iterator i = vec_render_models.begin(); i != vec_render_models.end(); i++)
	// 	{
	// 		if (!stricmp((*i)->GetName().c_str(), pchRenderModelName))
	// 		{
	// 			pRenderModel = *i;
	// 			break;
	// 		}
	// 	}

	// 	// load the model if we didn't find one
	// 	if (!pRenderModel)
	// 	{
	// 		vr::RenderModel_t *pModel;
	// 		vr::EVRRenderModelError error;
	// 		while (1)
	// 		{
	// 			error = vr::VRRenderModels()->LoadRenderModel_Async(pchRenderModelName, &pModel);
	// 			if (error != vr::VRRenderModelError_Loading)
	// 				break;

	// 			ThreadSleep(1);
	// 		}

	// 		if (error != vr::VRRenderModelError_None)
	// 		{
	// 			b3Printf("Unable to load render model %s - %s\n", pchRenderModelName, vr::VRRenderModels()->GetRenderModelErrorNameFromEnum(error));
	// 			return NULL; // move on to the next tracked device
	// 		}

	// 		vr::RenderModel_TextureMap_t *pTexture;
	// 		while (1)
	// 		{
	// 			error = vr::VRRenderModels()->LoadTexture_Async(pModel->diffuseTextureId, &pTexture);
	// 			if (error != vr::VRRenderModelError_Loading)
	// 				break;

	// 			ThreadSleep(1);
	// 		}

	// 		if (error != vr::VRRenderModelError_None)
	// 		{
	// 			b3Printf("Unable to load render texture id:%d for render model %s\n", pModel->diffuseTextureId, pchRenderModelName);
	// 			vr::VRRenderModels()->FreeRenderModel(pModel);
	// 			return NULL; // move on to the next tracked device
	// 		}

	// 		pRenderModel = new CGLRenderModel(pchRenderModelName);
	// 		if (!pRenderModel->BInit(*pModel, *pTexture))
	// 		{
	// 			b3Printf("Unable to create GL model from render model %s\n", pchRenderModelName);
	// 			delete pRenderModel;
	// 			pRenderModel = NULL;
	// 		}
	// 		else
	// 		{
	// 			vec_render_models.push_back(pRenderModel);
	// 		}
	// 		vr::VRRenderModels()->FreeRenderModel(pModel);
	// 		vr::VRRenderModels()->FreeTexture(pTexture);
	// 	}
	// 	return pRenderModel;
	// }

	// void setup_render_model_for_tracked_device(vr::TrackedDeviceIndex_t unTrackedDeviceIndex)
	// {
	// 	if (unTrackedDeviceIndex >= vr::k_unMaxTrackedDeviceCount)
	// 		return;

	// 	// try to find a model we've already set up
	// 	std::string sRenderModelName = get_tracked_device_string(hmd, unTrackedDeviceIndex, vr::Prop_RenderModelName_String);
	// 	CGLRenderModel *pRenderModel = find_or_load_render_model(sRenderModelName.c_str());
	// 	if (!pRenderModel)
	// 	{
	// 		std::string sTrackingSystemName = GetTrackedDeviceString(hmd, unTrackedDeviceIndex, vr::Prop_TrackingSystemName_String);
	// 		b3Printf("Unable to load render model for tracked device %d (%s.%s)", unTrackedDeviceIndex, sTrackingSystemName.c_str(), sRenderModelName.c_str());
	// 	}
	// 	else
	// 	{
	// 		m_rTrackedDeviceToRenderModel[unTrackedDeviceIndex] = pRenderModel;
	// 		m_rbShowTrackedDevice[unTrackedDeviceIndex] = true;
	// 	}
	// }

	// void setup_render_models()
	// {
	// 	memset(m_rTrackedDeviceToRenderModel, 0, sizeof(m_rTrackedDeviceToRenderModel));

	// 	if (!hmd)
	// 		return;

	// 	for (uint32_t unTrackedDevice = vr::k_unTrackedDeviceIndex_Hmd + 1; unTrackedDevice < vr::k_unMaxTrackedDeviceCount; unTrackedDevice++)
	// 	{
	// 		if (!hmd->IsTrackedDeviceConnected(unTrackedDevice))
	// 			continue;

	// 		SetupRenderModelForTrackedDevice(unTrackedDevice);
	// 	}
	// }


	bool create_all_shaders()
	{
		scene_program_id = CompileGLShader( 
			"Scene",

			// Vertex Shader
			"#version 410\n"
			"uniform mat4 matrix;\n"
			"layout(location = 0) in vec4 position;\n"
			"layout(location = 1) in vec2 v2UVcoordsIn;\n"
			"layout(location = 2) in vec3 v3NormalIn;\n"
			"out vec2 v2UVcoords;\n"
			"void main()\n"
			"{\n"
			"	v2UVcoords = v2UVcoordsIn;\n"
			"	gl_Position = matrix * position;\n"
			"}\n",

			// Fragment Shader
			"#version 410 core\n"
			"uniform sampler2D mytexture;\n"
			"in vec2 v2UVcoords;\n"
			"out vec4 outputColor;\n"
			"void main()\n"
			"{\n"
			"   outputColor = texture(mytexture, v2UVcoords);\n"
			"}\n"
			);
		scene_matrix_location = glGetUniformLocation( scene_program_id, "matrix" );
		if( scene_matrix_location == -1 )
		{
			b3Printf( "Unable to find matrix uniform in scene shader\n" );
			return false;
		}

		controller_transform_program_id = CompileGLShader(
			"Controller",

			// vertex shader
			"#version 410\n"
			"uniform mat4 matrix;\n"
			"layout(location = 0) in vec4 position;\n"
			"layout(location = 1) in vec3 v3ColorIn;\n"
			"out vec4 v4Color;\n"
			"void main()\n"
			"{\n"
			"	v4Color.xyz = v3ColorIn; v4Color.a = 1.0;\n"
			"	gl_Position = matrix * position;\n"
			"}\n",

			// fragment shader
			"#version 410\n"
			"in vec4 v4Color;\n"
			"out vec4 outputColor;\n"
			"void main()\n"
			"{\n"
			"   outputColor = v4Color;\n"
			"}\n"
			);
		controller_matrix_location = glGetUniformLocation( controller_transform_program_id, "matrix" );
		if( controller_matrix_location == -1 )
		{
			b3Printf( "Unable to find matrix uniform in controller shader\n" );
			return false;
		}

		render_model_program_id = CompileGLShader("render model",
			// vertex shader
			"#version 410\n"
			"uniform mat4 matrix;\n"
			"layout(location = 0) in vec4 position;\n"
			"layout(location = 1) in vec3 v3NormalIn;\n"
			"layout(location = 2) in vec2 v2TexCoordsIn;\n"
			"out vec2 v2TexCoord;\n"
			"void main()\n"
			"{\n"
			"	v2TexCoord = v2TexCoordsIn;\n"
			"	gl_Position = matrix * vec4(position.xyz, 1);\n"
			"}\n",

			//fragment shader
			"#version 410 core\n"
			"uniform sampler2D diffuse;\n"
			"in vec2 v2TexCoord;\n"
			"out vec4 outputColor;\n"
			"void main()\n"
			"{\n"
			"   outputColor = texture( diffuse, v2TexCoord);\n"
			"}\n"

			);
		render_model_matrix_location = glGetUniformLocation( render_model_program_id, "matrix" );
		if( render_model_matrix_location == -1 )
		{
			b3Printf( "Unable to find matrix uniform in render model shader\n" );
			return false;
		}

		lens_program_id = CompileGLShader(
			"Distortion",

			// vertex shader
			"#version 410 core\n"
			"layout(location = 0) in vec4 position;\n"
			"layout(location = 1) in vec2 v2UVredIn;\n"
			"layout(location = 2) in vec2 v2UVGreenIn;\n"
			"layout(location = 3) in vec2 v2UVblueIn;\n"
			"noperspective  out vec2 v2UVred;\n"
			"noperspective  out vec2 v2UVgreen;\n"
			"noperspective  out vec2 v2UVblue;\n"
			"void main()\n"
			"{\n"
			"	v2UVred = v2UVredIn;\n"
			"	v2UVgreen = v2UVGreenIn;\n"
			"	v2UVblue = v2UVblueIn;\n"
			"	gl_Position = position;\n"
			"}\n",

			// fragment shader
			"#version 410 core\n"
			"uniform sampler2D mytexture;\n"

			"noperspective  in vec2 v2UVred;\n"
			"noperspective  in vec2 v2UVgreen;\n"
			"noperspective  in vec2 v2UVblue;\n"

			"out vec4 outputColor;\n"

			"void main()\n"
			"{\n"
			"	float fBoundsCheck = ( (dot( vec2( lessThan( v2UVgreen.xy, vec2(0.05, 0.05)) ), vec2(1.0, 1.0))+dot( vec2( greaterThan( v2UVgreen.xy, vec2( 0.95, 0.95)) ), vec2(1.0, 1.0))) );\n"
			"	if( fBoundsCheck > 1.0 )\n"
			"	{ outputColor = vec4( 0, 0, 0, 1.0 ); }\n"
			"	else\n"
			"	{\n"
			"		float red = texture(mytexture, v2UVred).x;\n"
			"		float green = texture(mytexture, v2UVgreen).y;\n"
			"		float blue = texture(mytexture, v2UVblue).z;\n"
			"		outputColor = vec4( red, green, blue, 1.0  ); }\n"
			"}\n"
			);


		return scene_program_id != 0 
			&& controller_transform_program_id != 0
			&& render_model_program_id != 0
			&& lens_program_id != 0;
	}

	Matrix4 get_transform(vr::TrackedDeviceClass dev_class = vr::TrackedDeviceClass_Controller, int idx = 0) {
		Matrix4 m;
		int counter(0);
		for (uint32_t n = 0; n < vr::k_unMaxTrackedDeviceCount; n++)
			if (hmd->GetTrackedDeviceClass(n) == dev_class && tracked_device_pose[n].bPoseIsValid) {
				if (counter == idx) {
					m = tracked_device_pose_mat[n];
					break;
				}
				++idx;
			}
		return m;
	}

	void render_scene(vr::Hmd_Eye eye, vector<float> &m, vector<float> &m2) {
		//cout << "render scene proj mat: " << get_current_view_projection_matrix(eye) << endl;
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		//draw cubes
		/*
		glUseProgram(scene_program_id);
		glUniformMatrix4fv(scene_matrix_location, 1, GL_FALSE, get_current_view_projection_matrix(eye).get());
		glBindVertexArray(scene_vao);
		glBindTexture(GL_TEXTURE_2D, cube_texture);
		glDrawArrays(GL_TRIANGLES, 0, scene_vert_count);
		glBindVertexArray(0);*/

		//draw other cube
		int track_device(0);

		for (uint32_t n = 0; n < vr::k_unMaxTrackedDeviceCount; n++)
			if (hmd->GetTrackedDeviceClass(n) == vr::TrackedDeviceClass_Controller && tracked_device_pose[n].bPoseIsValid) {
				glUseProgram(scene_program_id);

				//if (tracked_device_pose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
				const Matrix4 &device_mat = tracked_device_pose_mat[n];
				Matrix4 matMVP = get_current_view_projection_matrix(eye) * device_mat;
				//cout << "found a controller: " << n << endl;

				//glUniformMatrix4fv( render_model_matrix_location, 1, GL_FALSE, matMVP.get() );
				glUniformMatrix4fv(scene_matrix_location, 1, GL_FALSE, matMVP.get());

				glUseProgram(scene_program_id);
				glBindVertexArray(cube_vao);
				glBindTexture(GL_TEXTURE_2D, cube_texture);
				glDrawArrays(GL_TRIANGLES, 0, cube_vert_count);
				glBindVertexArray(0);

				glUseProgram(0);
			}

		//Draw another cube
		glUseProgram(scene_program_id);

		Matrix4 fall_cube(&m[0]);
		Matrix4 matMVP = get_current_view_projection_matrix(eye) * fall_cube;
		glUniformMatrix4fv(scene_matrix_location, 1, GL_FALSE, matMVP.get());

		glUseProgram(scene_program_id);
		glBindVertexArray(cube_vao);
		glBindTexture(GL_TEXTURE_2D, cube_texture);
		glDrawArrays(GL_TRIANGLES, 0, cube_vert_count);
		glBindVertexArray(0);


		Matrix4 fall_cube2(&m2[0]);
		Matrix4 matMVP2 = get_current_view_projection_matrix(eye) * fall_cube2;
		glUniformMatrix4fv(scene_matrix_location, 1, GL_FALSE, matMVP2.get());

		glUseProgram(scene_program_id);
		glBindVertexArray(cube_vao);
		glBindTexture(GL_TEXTURE_2D, cube_texture);
		glDrawArrays(GL_TRIANGLES, 0, cube_vert_count);
		glBindVertexArray(0);

		glUseProgram(0);
	}


	Matrix4 get_current_view_projection_matrix( vr::Hmd_Eye eye )
	{
		Matrix4 matMVP;
		if( eye == vr::Eye_Left )
		{
			matMVP = mat4_projection_left * mat4_eye_pos_left * mat4_hmd_pose;
		}
		else if( eye == vr::Eye_Right )
		{
			matMVP = mat4_projection_right * mat4_eye_pos_right *  mat4_hmd_pose;
		}

		return matMVP;
	}


	bool create_frame_buffer( int width, int height, FramebufferDesc &framebufferDesc)
	{
		glGenFramebuffers(1, &framebufferDesc.m_nRenderFramebufferId );
		glBindFramebuffer(GL_FRAMEBUFFER, framebufferDesc.m_nRenderFramebufferId);

		glGenRenderbuffers(1, &framebufferDesc.m_nDepthBufferId);
		glBindRenderbuffer(GL_RENDERBUFFER, framebufferDesc.m_nDepthBufferId);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT, width, height );
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,	framebufferDesc.m_nDepthBufferId );

		glGenTextures(1, &framebufferDesc.m_nRenderTextureId );
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, framebufferDesc.m_nRenderTextureId );
		glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, width, height, true);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, framebufferDesc.m_nRenderTextureId, 0);

		glGenFramebuffers(1, &framebufferDesc.m_nResolveFramebufferId );
		glBindFramebuffer(GL_FRAMEBUFFER, framebufferDesc.m_nResolveFramebufferId);

		glGenTextures(1, &framebufferDesc.m_nResolveTextureId );
		glBindTexture(GL_TEXTURE_2D, framebufferDesc.m_nResolveTextureId );
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferDesc.m_nResolveTextureId, 0);

		// check FBO status
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE)
		{
			return false;
		}

		glBindFramebuffer( GL_FRAMEBUFFER, 0 );

		return true;
	}

	void setup_cameras()
	{
		mat4_projection_left = get_hmd_matrix_projection_eye(vr::Eye_Left);
		mat4_projection_right = get_hmd_matrix_projection_eye(vr::Eye_Right);
		mat4_eye_pos_left = get_hmd_matrix_pose_eye(vr::Eye_Left);
		mat4_eye_pos_right = get_hmd_matrix_pose_eye(vr::Eye_Right);
	}

	bool setup_stereo_render_targets()
	{
		if (!hmd)
			return false;

		hmd->GetRecommendedRenderTargetSize(&render_width, &render_height);

		create_frame_buffer(render_width, render_height, left_eye_desc);
		create_frame_buffer(render_width, render_height, right_eye_desc);

		left_frame_data.resize(render_width * render_height * 4);
		right_frame_data.resize(render_width * render_height * 4);

		return true;
	}


	void render_stereo_targets(vector<float> &m, vector<float> &m2) {
		glClearColor( 0.15f, 0.15f, 0.18f, 1.0f ); // nice background color, but not black
		glEnable( GL_MULTISAMPLE );

		app->m_instancingRenderer->init();
		Matrix4 rotYtoZ = rotYtoZ.identity();

		//left eye
		{
			/*
			Matrix4 view_mat_left = mat4_eye_pos_left * mat4_hmd_pose * rotYtoZ;
			Matrix4 view_mat_center = mat4_hmd_pose * rotYtoZ;
			//Matrix4 view_mat_left =  rotYtoZ;
			//Matrix4 view_mat_center = rotYtoZ;

			Matrix4 m = view_mat_center;
			const float* mat = m.invertAffine().get();

			float dist=1;

			glBindFramebuffer( GL_FRAMEBUFFER, left_eye_desc.m_nRenderFramebufferId);
 			glViewport(0, 0, render_width, render_height);

 			///////////*****  Set Camera
	 	// 				float dist=1;
			app->m_instancingRenderer->getActiveCamera()->setCameraTargetPosition(
				mat[12]-dist*mat[8],
				mat[13]-dist*mat[9],
				mat[14]-dist*mat[10]
				);
			*/
			///////
			//cout << "is vr: " << app->m_instancingRenderer->getActiveCamera()->isVRCamera() << endl;
			
			//app->m_instancingRenderer->getActiveCamera()->setCameraUpVector(mat[0], mat[1], mat[2]);
			//app->m_instancingRenderer->getActiveCamera()->setVRCamera(view_mat_left.get(), mat4_projection_left.get());

			//app->m_instancingRenderer->updateCamera(app->getUpAxis());

			glBindFramebuffer( GL_FRAMEBUFFER, left_eye_desc.m_nRenderFramebufferId );
		 	glViewport(0, 0, render_width, render_height );

			//app->m_window->startRendering();
			//app->m_instancingRenderer->setRenderFrameBuffer((unsigned int)left_eye_desc.m_nRenderFramebufferId);
			render_scene( vr::Eye_Left, m, m2 );

			//read to cpu buffer
			glReadPixels(0, 0, render_width, render_height, GL_RGBA, GL_UNSIGNED_BYTE, &left_frame_data[0]);

			//other stuff
			glBindFramebuffer( GL_FRAMEBUFFER, 0 );
			glDisable( GL_MULTISAMPLE );
			 	
		 	glBindFramebuffer(GL_READ_FRAMEBUFFER, left_eye_desc.m_nRenderFramebufferId);
		    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, left_eye_desc.m_nResolveFramebufferId );

		    glBlitFramebuffer( 0, 0, render_width, render_height, 0, 0, render_width, render_height,
				GL_COLOR_BUFFER_BIT,
		 		GL_LINEAR );

		 	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
		    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0 );	

			glEnable( GL_MULTISAMPLE );
		}

		{ //Right eye
			glBindFramebuffer( GL_FRAMEBUFFER, right_eye_desc.m_nRenderFramebufferId );
		 	glViewport(0, 0, render_width, render_height );

			//app->m_window->startRendering();
			//app->m_instancingRenderer->setRenderFrameBuffer((unsigned int)right_eye_desc.m_nRenderFramebufferId);
			render_scene( vr::Eye_Right, m, m2 );

			//read to cpu buffer
			glReadPixels(0, 0, render_width, render_height, GL_RGBA, GL_UNSIGNED_BYTE, &right_frame_data[0]);

			//other stuff
			glBindFramebuffer( GL_FRAMEBUFFER, 0 );
			glDisable( GL_MULTISAMPLE );
			 	
		 	glBindFramebuffer(GL_READ_FRAMEBUFFER, right_eye_desc.m_nRenderFramebufferId);
		    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, right_eye_desc.m_nResolveFramebufferId );

		    glBlitFramebuffer( 0, 0, render_width, render_height, 0, 0, render_width, render_height,
				GL_COLOR_BUFFER_BIT,
		 		GL_LINEAR );

		 	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
		    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0 );	

			glEnable( GL_MULTISAMPLE );
		}
	}

	void render_frame(vector<float> &m, vector<float> &m2) {
		if (hmd) {
			//cout << "rendering and submitting" << endl;
	 		render_stereo_targets(m, m2);

	 		vr::Texture_t left_eye_texture = {(void*)left_eye_desc.m_nResolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
			vr::VRCompositor()->Submit(vr::Eye_Left, &left_eye_texture );
			vr::Texture_t right_eye_texture = {(void*)right_eye_desc.m_nResolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
			vr::VRCompositor()->Submit(vr::Eye_Right, &right_eye_texture );

			// Clear
			{
				B3_PROFILE("glClearColor");
				// We want to make sure the glFinish waits for the entire present to complete, not just the submission
				// of the command. So, we do a clear here right here so the glFinish will wait fully for the swap.
				glClearColor( 0, 0, 0, 1 );
				glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
			}

			// Flush and wait for swap.
			bool blank = true;
			if (blank) {
				B3_PROFILE("glFlushglFinish");

				glFlush();
				glFinish();
			}
	 	}
	}

	vr::IVRSystem *hmd;
	vr::IVRRenderModels *render_models;

	vr::TrackedDevicePose_t tracked_device_pose[vr::k_unMaxTrackedDeviceCount];
	Matrix4 tracked_device_pose_mat[vr::k_unMaxTrackedDeviceCount];

	vr::VRControllerState_t prev_controller_states[vr::k_unMaxTrackedDeviceCount];

	int window_height = 0, window_width = 0;
	float near_clip, far_clip, fscale, fscale_spacing;

	std::string driver_str, display_str;

	Matrix4 mat4_hmd_pose;
	Matrix4 mat4_eye_pos_left;
	Matrix4 mat4_eye_pos_right;

	Matrix4 mat4_projection_center;
	Matrix4 mat4_projection_left;
	Matrix4 mat4_projection_right;


	char dev_class_char[vr::k_unMaxTrackedDeviceCount];   // for each device, a character representing its class

	//Matrix4 rmat_device_pose[vr::k_unMaxTrackedDeviceCount];


	FramebufferDesc left_eye_desc;
	FramebufferDesc right_eye_desc;

	int valid_pose_count = 0;
	string str_pose_classes;

		

	GLint scene_matrix_location = 0;
	GLint controller_matrix_location = 0;
	GLint render_model_matrix_location = 0;

	GLuint scene_program_id = 0, controller_transform_program_id = 0, render_model_program_id = 0, lens_program_id = 0;

	GLuint scene_vert_buffer = 0, scene_vao = 0;
	unsigned int scene_vert_count;
	
	GLuint cube_vert_buffer = 0, cube_vao = 0;
	unsigned int cube_vert_count = 0;

	GLuint cube_texture = 0;
	GLuint lens_vertex_array = 0;
	GLuint gl_id_index_buffer = 0;
	GLuint gl_id_vertex_buffer = 0;
	unsigned int ui_index_size = 0;

	float scene_volume_dim;

	uint32_t render_width = 0, render_height = 0;
	//vector<CGLRenderModel*> vec_render_models;

	vector<uint8_t> left_frame_data, right_frame_data;

	SimpleOpenGL3App* app = 0;
};
/*
inline void disable_vsync() {
  //request disable VSYNC
	typedef bool (APIENTRY *PFNWGLSWAPINTERVALFARPROC)(int);
	PFNWGLSWAPINTERVALFARPROC wglSwapIntervalEXT = 0;
	wglSwapIntervalEXT = 
		(PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");
	if (wglSwapIntervalEXT)
		wglSwapIntervalEXT(0);
		}*/

void print(btVector3 v) {
	cout << v.getX() << "," << v.getY() << "," << v.getZ() << endl;
}


struct PID {
	bool init = false;

	float decay = .70;
	float Ke = 25;
	float Kde = 20.;
	float Kie = 10.;
	float amp = 1;

	btVector3 ie, last_e;

	PID() {
		ie.setZero();
		last_e.setZero();
	}

	btVector3 update(btVector3 err, float dt) {
		if (!init) { last_e = err; init = true; }

		//cout << "err: ";
		//print(err);

		ie += err * dt;
		ie *= pow(decay, 1.0 / dt);

		btVector3 de(err);
		de -= last_e;

		//cout << "dt: " << dt << endl;
		//print(de);
		de /= dt;
		last_e = err;
		//cout << "err, de, ie: " << endl;
		//print(err);
		//print(de);
		//print(ie);
		btVector3 effort = err * Ke + de * Kde + ie * Kie;
		return effort * amp;
	}

};


int main(int argc, char **argv) {
	DB db("exp.db");

	GUI gui;

	cout << "VRL" << endl;
	b3CommandLineArgs args(argc, argv);
	

	Timer timer;
	Physics physics;
	VRSetup vr_setup;
	vr_setup.setup_scene();
	vr_setup.setup_cube();
	//disable_vsync();

	PID pid_control;
	PID pid_link;
	float link_distance(.5);

	uint64_t frame_counter(0);

	cout << "starting shit" << endl;
	while (true) {
		float dt = timer.since();
		timer.reset();
		auto t = vr_setup.get_transform();
		cout << "trans: " << t << endl;

		btTransform trans;
		physics.fallRigidBody->getMotionState()->getWorldTransform(trans);
		vector<btScalar> m(16);
		trans.getOpenGLMatrix(&m[0]);
		vector<float> mf(m.begin(), m.end());

		physics.fallRigidBody2->getMotionState()->getWorldTransform(trans);
		vector<btScalar> m2(16);
		trans.getOpenGLMatrix(&m2[0]);
		vector<float> mf2(m2.begin(), m2.end());

		//cout << "controller: " << t[12] << "," << t[13] << "," << t[14] << endl;
		cout << "object: " << m[12] << "," << m[13] << "," << m[14] << endl;
		{
			btVector3 e(t[12] - m[12], t[13] - m[13], t[14] - m[14]);
			btVector3 effort = pid_control.update(e, dt);
			//print(effort);
			physics.fallRigidBody->applyCentralForce(effort);
		}
		{
			btVector3 diff(m2[12] - m[12], m2[13] - m[13], m2[14] - m[14]);
			float norm = diff.norm();
			btVector3 norm_diff = diff / (norm + .0001);
			float dist = link_distance - norm;
			btVector3 e = norm_diff * dist;

			btVector3 effort = pid_link.update(e, dt);
			float proj_effort = effort.dot(norm_diff);

			physics.fallRigidBody2->applyCentralForce(norm_diff * proj_effort);

		}

		physics.step(dt);

		vr_setup.update_hmd_matrix_pose();
		vr_setup.render_frame(mf, mf2);
		//vr_setup.handle_vr_input(&vr_controller_input_callback_printer, &vr_controller_move_callback_printer);
		vr_setup.handle_vr_input(0, 0);

		
		// Store frame
		WriteMessage msg;
		auto b = msg.builder<Frame>();
		
		Bytes img(&vr_setup.right_frame_data[0], vr_setup.right_frame_data.size()); //supposed image
		b.setPixels(img.kjp());
		b.setTime(now_ms());
		int n_objects(3);
		auto objects = b.initObjects(n_objects);
		
		//objects[0].setTransform();

		auto data = msg.bytes();
		Bytes id(8);
		copy(reinterpret_cast<char*>(&frame_counter), reinterpret_cast<char*>(&frame_counter) + 8, id.ptr<char*>());
		db.put(id, data);

		//vr_setup.setup_scene();
		//Sleep(100);
		++frame_counter;
	}
}

