using Cxx = import "/capnp/c++.capnp";
@0xa3ec9eb77ec1fa59;
$Cxx.namespace("cap");
    
    
struct Quat {
    a @0 : Float32;
    b @1 : Float32;
    c @2 : Float32;
    d @3 : Float32;
}

struct Pos {
    x @0 : Float32;
    y @1 : Float32;
    z @2 : Float32;
}

struct Snap {
    objectIds @0 : List(UInt64);
    variableIds @1 : List(UInt64);
    triggerIds @2 : List(UInt64);   

    timestamp @3 : UInt64;
    reward @4 : Float32;
}

struct Variable {
    nameId @0 : UInt32;
    val @1 : Float32;
    union {
        distance @2 : NamePair;
        free @3 : Float32;
        mark @4 : Float32;
    }
}

struct NamePair {
    nameId1 @0 : UInt32;
    nameId2 @1 : UInt32;
}

struct NameLimit {
    nameId @0 : UInt32;
    limit @1 : Float32;
}
        
struct Trigger {
    nameId @0 : UInt32;
    functionNameId @1 : UInt32;
    union {
        limit @2 : NameLimit;
        click @3 : Void;
        inBox @4 : NamePair;
        next @5 : Void;
    }
}

    
struct Recording {
    objects @0 : List(Object);
    variables @1 : List(Variable);
    triggers @2 : List(Trigger);
    names @3 : List(Text);

    snaps @4 : List(Snap);
}

struct Controller {
    right @0 : Bool;
    clicked @1 : Bool;
    pressed @2 : Bool;
}

struct Box {
    w @0 : Float32;
    h @1 : Float32;
    d @2 : Float32;
    texture @3 : Text;
}
        
struct Object {
    pos @0 : Pos;
    quat @1 : Quat;
    nameId @2 : UInt32;


    union {
        hmd @3 : Void;
        controller @4 : Controller;
        point @5 : Void;
        canvas @6 : Text;
        box @7 : Box;
    }
}

