#ifndef __PHYSICS_H__
#define __PHYSICS_H__

#include <Bullet3Common/b3Quaternion.h>
#include <Bullet3Common/b3Transform.h>
#include <Bullet3Common/b3CommandLineArgs.h>
#include <btBulletDynamicsCommon.h>
#include <BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h>
#include <BulletCollision/CollisionShapes/btStaticPlaneShape.h>

#include <algorithm>
#include <vector>
#include <iostream>

#include "Matrices.h"
#include "utils.h"

struct Physics {
	Physics() {
		broadphase = new btDbvtBroadphase();
		collision_configuration = new btDefaultCollisionConfiguration();
		dispatcher = new btCollisionDispatcher(collision_configuration);
		solver = new btSequentialImpulseConstraintSolver;
		dynamics_world = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collision_configuration);

		dynamics_world->setGravity(btVector3(0, -9.8, 0));

		init_objects();
	}

	void init_objects() {
		//ground
		groundShape = new btStaticPlaneShape(btVector3(0, 1, 0), 1);
		fallShape = new btSphereShape(.1);
		fallShape2 = new btSphereShape(.1);
		groundMotionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, -1, 0)));
		btRigidBody::btRigidBodyConstructionInfo groundRigidBodyCI(0, groundMotionState, groundShape, btVector3(0, 0, 0));

        btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
        dynamics_world->addRigidBody(groundRigidBody);

        //object
		fallMotionState =
			new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0.4, 2.1, 0)));
        fallMotionState2 =
                new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0.0, 3.2, 0)));
        btScalar mass = 1;
        btVector3 fallInertia(0, 0, 0);
        fallShape->calculateLocalInertia(mass, fallInertia);
		fallShape2->calculateLocalInertia(mass, fallInertia);

        btRigidBody::btRigidBodyConstructionInfo fallRigidBodyCI(mass, fallMotionState, fallShape, fallInertia);
       	fallRigidBody = new btRigidBody(fallRigidBodyCI);
        dynamics_world->addRigidBody(fallRigidBody);

        btRigidBody::btRigidBodyConstructionInfo fallRigidBodyCI2(mass, fallMotionState2, fallShape2, fallInertia);
		fallRigidBodyCI2.m_linearDamping = .1;
       	fallRigidBody2 = new btRigidBody(fallRigidBodyCI2);
        dynamics_world->addRigidBody(fallRigidBody2);
				
		fallRigidBody->setActivationState(DISABLE_DEACTIVATION);
		fallRigidBody2->setActivationState(DISABLE_DEACTIVATION);
	}

	~Physics() {
		delete dynamics_world;
	    delete solver;
	    delete dispatcher;
	    delete collision_configuration;
	    delete broadphase;
	}

	void step(float dt) {
		//btVector3 force(-0.01, 0, .0), ref(0, 0, 1);
		//fallRigidBody->applyCentralForce(force);
		int max_sub = 10;
		dynamics_world->stepSimulation(dt, max_sub);

        // btTransform trans;
        // fallRigidBody->getMotionState()->getWorldTransform(trans);
	}

	std::vector<float> body_to_vec(btRigidBody *body) {
		auto lin_vel = fallRigidBody->getLinearVelocity();
		auto ang_vel = fallRigidBody->getAngularVelocity();

		std::vector<float> vals(3 + 3 + 16);
		float *val_it(&vals[0]);
		std::copy(&ang_vel.m_floats[0], &ang_vel.m_floats[3], val_it);
		val_it += 3;
		std::copy(&lin_vel.m_floats[0], &lin_vel.m_floats[3], val_it);
		val_it += 3;

		btTransform trans;
		fallRigidBody->getMotionState()->getWorldTransform(trans);
		std::vector<btScalar> m(16);
		trans.getOpenGLMatrix(&m[0]);
		std::copy(&m[0], &m[16], val_it);
		return vals;
	}

	void body_from_vec(btRigidBody *body, std::vector<float> &data) {
		btVector3 lin_vel;
		btVector3 ang_vel;

		std::copy(&data[0], &data[3], &lin_vel[0]);
		std::copy(&data[3], &data[6], &ang_vel[0]);

		fallRigidBody->setLinearVelocity(lin_vel);
		fallRigidBody->setAngularVelocity(ang_vel);

		std::vector<btScalar> m(16);
		btTransform trans;
		std::copy(&data[6], &data[6 + 16], &m[0]);
		trans.setFromOpenGLMatrix(&m[0]);
		fallRigidBody->getMotionState()->setWorldTransform(trans);

	}

	btBroadphaseInterface* broadphase;
	btDefaultCollisionConfiguration* collision_configuration;
	btCollisionDispatcher* dispatcher;
	btSequentialImpulseConstraintSolver* solver;
	btDiscreteDynamicsWorld* dynamics_world;

	//Objects
	std::vector<btCollisionShape*> collision_shapes;
	std::vector<btSliderConstraint*> slider_constraints;

	btCollisionShape* groundShape;
	btCollisionShape *fallShape, *fallShape2;
	btDefaultMotionState* groundMotionState;
	btRigidBody* groundRigidBody;


	btDefaultMotionState *fallMotionState, *fallMotionState2;
	btRigidBody *fallRigidBody, *fallRigidBody2;


};


#endif
