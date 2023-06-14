#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z12createCameraii(int, int);
extern void __device_stub__Z8setupRNGP17curandStateXORWOWii(curandState *, int, int);
extern void __device_stub__Z11createScenePP11rayHittablePP5scene(struct rayHittable **, struct scene **);
extern void __device_stub__Z6renderP4vec3iiPP5sceneP17curandStateXORWOW(struct vec3 *, int, int, struct scene **, curandState *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void);
#pragma section(".CRT$XCT",read)
__declspec(allocate(".CRT$XCT"))static void (*__dummy_static_init__sti____cudaRegisterAll[])(void) = {__sti____cudaRegisterAll};
void __device_stub__Z12createCameraii(
int __par0, 
int __par1)
{
__cudaLaunchPrologue(2);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 4Ui64);
__cudaLaunch(((char *)((void ( *)(int, int))createCamera)));
}
void createCamera( int __cuda_0,int __cuda_1)
{__device_stub__Z12createCameraii( __cuda_0,__cuda_1);
}
#line 1 "x64/Debug/main.cudafe1.stub.c"
void __device_stub__Z8setupRNGP17curandStateXORWOWii(
curandState *__par0, 
int __par1, 
int __par2)
{
__cudaLaunchPrologue(3);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 12Ui64);
__cudaLaunch(((char *)((void ( *)(curandState *, int, int))setupRNG)));
}
void setupRNG( curandState *__cuda_0,int __cuda_1,int __cuda_2)
{__device_stub__Z8setupRNGP17curandStateXORWOWii( __cuda_0,__cuda_1,__cuda_2);
}
#line 1 "x64/Debug/main.cudafe1.stub.c"
void __device_stub__Z11createScenePP11rayHittablePP5scene(
struct rayHittable **__par0, 
struct scene **__par1)
{
__cudaLaunchPrologue(2);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaLaunch(((char *)((void ( *)(struct rayHittable **, struct scene **))createScene)));
}
void createScene( struct rayHittable **__cuda_0,struct scene **__cuda_1)
{__device_stub__Z11createScenePP11rayHittablePP5scene( __cuda_0,__cuda_1);
}
#line 1 "x64/Debug/main.cudafe1.stub.c"
void __device_stub__Z6renderP4vec3iiPP5sceneP17curandStateXORWOW(
struct vec3 *__par0, 
int __par1, 
int __par2, 
struct scene **__par3, 
curandState *__par4)
{
__cudaLaunchPrologue(5);
__cudaSetupArgSimple(__par0, 0Ui64);
__cudaSetupArgSimple(__par1, 8Ui64);
__cudaSetupArgSimple(__par2, 12Ui64);
__cudaSetupArgSimple(__par3, 16Ui64);
__cudaSetupArgSimple(__par4, 24Ui64);
__cudaLaunch(((char *)((void ( *)(struct vec3 *, int, int, struct scene **, curandState *))render)));
}
void render( struct vec3 *__cuda_0,int __cuda_1,int __cuda_2,struct scene **__cuda_3,curandState *__cuda_4)
{__device_stub__Z6renderP4vec3iiPP5sceneP17curandStateXORWOW( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
}
#line 1 "x64/Debug/main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback(
void **__T151)
{
__nv_dummy_param_ref(__T151);
__nv_save_fatbinhandle_for_managed_rt(__T151);
__cudaRegisterEntry(__T151, ((void ( *)(struct vec3 *, int, int, struct scene **, curandState *))render), _Z6renderP4vec3iiPP5sceneP17curandStateXORWOW, (-1));
__cudaRegisterEntry(__T151, ((void ( *)(struct rayHittable **, struct scene **))createScene), _Z11createScenePP11rayHittablePP5scene, (-1));
__cudaRegisterEntry(__T151, ((void ( *)(curandState *, int, int))setupRNG), _Z8setupRNGP17curandStateXORWOWii, (-1));
__cudaRegisterEntry(__T151, ((void ( *)(int, int))createCamera), _Z12createCameraii, (-1));
__cudaRegisterVariable(__T151, __shadow_var(precalc_xorwow_matrix,::precalc_xorwow_matrix), 0, 102400Ui64, 0, 0);
__cudaRegisterVariable(__T151, __shadow_var(precalc_xorwow_offset_matrix,::precalc_xorwow_offset_matrix), 0, 102400Ui64, 0, 0);
__cudaRegisterVariable(__T151, __shadow_var(mrg32k3aM1,::mrg32k3aM1), 0, 2304Ui64, 0, 0);
__cudaRegisterVariable(__T151, __shadow_var(mrg32k3aM2,::mrg32k3aM2), 0, 2304Ui64, 0, 0);
__cudaRegisterVariable(__T151, __shadow_var(mrg32k3aM1SubSeq,::mrg32k3aM1SubSeq), 0, 2016Ui64, 0, 0);
__cudaRegisterVariable(__T151, __shadow_var(mrg32k3aM2SubSeq,::mrg32k3aM2SubSeq), 0, 2016Ui64, 0, 0);
__cudaRegisterVariable(__T151, __shadow_var(mrg32k3aM1Seq,::mrg32k3aM1Seq), 0, 2304Ui64, 0, 0);
__cudaRegisterVariable(__T151, __shadow_var(mrg32k3aM2Seq,::mrg32k3aM2Seq), 0, 2304Ui64, 0, 0);
__cudaRegisterVariable(__T151, __shadow_var(__cr_lgamma_table,::__cr_lgamma_table), 0, 72Ui64, 1, 0);
__cudaRegisterVariable(__T151, __shadow_var(dCam,::dCam), 0, 48Ui64, 0, 0);
}
static void __sti____cudaRegisterAll(void)
{
__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);
}
