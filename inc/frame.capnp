@0xa63f96ee0faf2eeb;

enum Type {
	hmd @0;
	controllerLeft @1;
	controllerRight @2;
	ball @3;
}

struct Obj {
  id @0 :UInt64;
	type @1 :Type;
	transform @2 :Data;
	momentum @3 :Data;
	extra @4 :Data;
}

struct Frame {
   time @0 :UInt16;
   pixels @1 :Data;
   state @2 :Data;
   action @3 :Data;
   obs @4 :Data;
   objects @5 :List(Obj);
}

struct Experiment {
  name @0 :Text;
  type @1 :Text;
  description @2 :Text;
}
