#ifndef __FAST_PIPATH_OPENING_T_HPP__
#define __FAST_PIPATH_OPENING_T_HPP__

#include <string.h>

#include <stdlib.h>

#include <stdio.h>

#include <queue>

#include <limits>


// Parsimonious incomplete-path openings
// Author : Petr Dokladal (based on V. Morard's code)
// implements Parsimonious Incomplete-Path Openings using the Rank Opening (see Pierre Soille, On morphological operators based on rank filters, Pattern Recognition 35 (2002) 527-535)

// the rank opening is implemented as a rank filter followed by a dilation. It uses histograms.

// TO DOS : 1) make it isotropic. 
//          2) extend the border mirroring to SE>n. Will be useful for short diagonal paths in the corners of the image 
//             (see how this extension can be done)





namespace morphee
{
  namespace FastPathOpening
  {
		

    typedef unsigned char pixtype;
    typedef unsigned int offset_t;
    typedef double weight_t;

#define MAXOF(x,y) ((x)>(y)?(x):(y))
#define MINOF(x,y) ((x)<(y)?(x):(y))

    using namespace std;
    int MIRROR_BORDERS = 0;


    pixtype rank (pixtype* H, int r)
    {
      int cnt = 0;  
      pixtype ii;
      int HWidth = numeric_limits<pixtype>::max();
      for (ii = 0; ii<HWidth; ii++)
	{
	  cnt += H[ii];
	  if (cnt > r)
	    break;
	}
      return ii;
    }




    int rank_filter_indx (pixtype *x, offset_t *indx, int n, int SE, int r, pixtype *y /*, weight_t *w*/)
    {
      int HWidth = numeric_limits<pixtype>::max();
      pixtype* H = (pixtype*) calloc (HWidth, sizeof(pixtype));
      //weight_t Hsum;
  
      //  pad border 
      offset_t* indx_pad = (offset_t*) calloc (n+2*SE, sizeof(offset_t));
      //weight_t* w_pad = (weight_t*) calloc (n+2*SE, sizeof(weight_t));
      memcpy (indx_pad+SE, indx, n*sizeof(offset_t));
      //memcpy (w_pad+SE, w, n*sizeof(weight_t));
      switch (MIRROR_BORDERS) {
      case 0: {
	//  pad border by replicating
	for (int ii=0; ii<SE; ii++) {
	  indx_pad[ii] = indx[0];
	  indx_pad[n+SE+ii] = indx[n-1];
	  /*w_pad[ii] = w[0];
	    w_pad[n+SE+ii] = w[n-1];*/ }
	break; }
      case 1: {
	//  pad border by mirroring
	for (int ii=0; ii<SE; ii++) {
	  indx_pad[ii] = indx[SE-ii];
	  indx_pad[n+ii+SE] = indx[n-2-ii];
	  /*w_pad[ii] = w[SE-ii];
	    w_pad[n+ii+SE] = w[n-2-ii];*/ }
	break; }
      }
  
      // init histogram
      for (int ii=0; ii<SE; ii++) {
	H[x[indx_pad[ii+SE-SE/2]]] += 1;        // no weight is used
	/*H[x[indx_pad[ii+SE-SE/2]]] += w[ii+SE-SE/2]; 
	  Hsum += w_pad[ii+SE-SE/2-1]; */ }
      y[0] = rank(H,r);
  
      for (int ii=0; ii<n-1; ii++)
	{
	  H[x[indx_pad[ii+SE-SE/2]]] -= 1;      // no weight is used
	  H[x[indx_pad[ii+2*SE-SE/2]]] += 1;    // no weight is used
	  /*H[x[indx_pad[ii+SE-SE/2]]] -= w[ii+SE-SE/2];
	    H[x[indx_pad[ii+2*SE-SE/2]]] += w[ii+2*SE-SE/2];
	    Hsum -= w_pad[ii+SE-SE/2-1];
	    Hsum += w_pad[ii+2*SE-SE/2-1];
	    int SEend = ii+2*SE-SE/2;
	    int SEstart = ii+SE-SE/2;
	    weight_t w_in = w_pad[ii+2*SE-SE/2-1];
	    weight_t w_out = w_pad[ii+SE-SE/2-1];*/
	  y[ii+1] = rank (H, r);
	}
  
      free (indx_pad);
      free (H);
      return 0;
    }





    int conj_dilation (pixtype *x, int n, int SE, pixtype *y)
    {
      int HWidth = numeric_limits<pixtype>::max();
      pixtype* H = (pixtype*) calloc (HWidth, sizeof(pixtype));
  
      //  pad border by replicating
      pixtype* x_pad = (pixtype*) calloc (n+2*SE, sizeof(pixtype));
      memcpy (x_pad+SE, x, n*sizeof(pixtype));
      memset (x_pad, x[0], SE);
      memset (x_pad+n+SE, x[n-1], SE);

      // init histogram
      for (int ii=0; ii<SE; ii++)
	H[x_pad[ii+SE-SE/2+(SE+1)%2]] += 1;
      y[0] = rank(H,SE-1);
  
      for (int ii=0; ii<n-1; ii++)
	{
	  H[x_pad[ii+SE-SE/2+(SE+1)%2]] -= 1;
	  H[x_pad[ii+2*SE-SE/2+(SE+1)%2]] += 1;
	  y[ii+1] = rank (H, SE-1);
	}
  
      free (x_pad);
      free (H);
      return 0;
    }









    int rank_open (pixtype *x, offset_t *indx, int n, int SE, int r, pixtype *y /*, weight_t *w */)
    {
      pixtype* xi = (pixtype*) calloc (n, sizeof(pixtype));
      pixtype* dxi = (pixtype*) calloc (n, sizeof(pixtype));
      
      rank_filter_indx (x, indx, n, SE, r, xi);
      conj_dilation (xi, n, SE, dxi);
  
      for (int ii=0; ii<n; ii++)
	y[indx[ii]] = MAXOF (y[indx[ii]], MINOF (x[indx[ii]], dxi[ii]));
  
      free (xi);
      free (dxi);
      return 0;
    }







    template<class T1, class T2> 
    int  t_ImParsimoniousIncompletePathOpening (Image<T1> & imIn, int Size, int tolerance, int step, int rec, Image<T2> & imOut)
    {
      if( tolerance<0 ){
	MORPHEE_REGISTER_ERROR("tolerance must be >=0");
	return RES_ERROR_BAD_ARG;
      }
      
      //Check inputs
      if( ! imIn.isAllocated() || ! imOut.isAllocated() ){
	MORPHEE_REGISTER_ERROR("Image not allocated");
	return RES_NOT_ALLOCATED;
      }
      if(!t_CheckWindowSizes(imIn, imOut)){
	MORPHEE_REGISTER_ERROR("Bad window sizes");
	return RES_ERROR_BAD_WINDOW_SIZE;
      }
      if(step<=0){
	MORPHEE_REGISTER_ERROR("Bad arg step must be >=1");
	return RES_ERROR_BAD_ARG;
      }

      size = Size;

      double lengthDir;
      int W,H,i,j,stackSize,whichLine,Dir, DirH,DirV;;
      Node<T2> MyStack[256];

      W=imIn.getWxSize();
      H=imIn.getWySize();
      T1 *bufferIn  = imIn.rawPointer();
      T1 F;
      T2 *bufferOut = imOut.rawPointer();
			
      //Initialisation
      memset(bufferOut,0,W*H*sizeof(T2));
      LineIdx = new int[W+H];
      if(LineIdx==0){
	MORPHEE_REGISTER_ERROR("LineIdx = new int[W+H];");
	return RES_ERROR_MEMORY;
      }
		
      offset_t *indx = new offset_t [W+H];


      //First direction Left to right
      //printf ("direction 1: left-to-right\n");
      for(j=0;j<H;j+=step){
	wp=0;
	Length=0;
	stackSize=0;
	indx[0] = j*W;
	whichLine = j;
				
	for(i=1;i<W;i++){
	  F = bufferIn[i+whichLine*W];
	  indx[i] = i+whichLine*W;
	  Dir = 0;
	  lengthDir=1;
	  if(whichLine-1 >= 0 && F < bufferIn[i+(whichLine-1)*W]){
	    F = bufferIn[i+(whichLine-1)*W];
	    indx[i] = i+(whichLine-1)*W;
	    Dir = -1;
	    lengthDir=SQRT_2;
	  }
	  if(whichLine+1 < H && F < bufferIn[i+(whichLine+1)*W]){
	    indx[i] = i+(whichLine+1)*W;
	    Dir = 1;
	    lengthDir=SQRT_2;
	  }
	  whichLine += Dir;
	}
	rank_open (bufferIn, indx, W, size, tolerance, bufferOut);
      }		
			


      //Second direction Right to left
      //printf ("direction 2: right-to-left\n");
      for(j=0;j<H;j+=step){
	wp=0;
	Length=0;
	stackSize=0;
	indx[W-1] = W-1+j*W;
	whichLine = j;
	for(i=W-2;i>=0;i--){
	  F = bufferIn[i+whichLine*W];
	  indx[i] = i+whichLine*W;
	  Dir = 0;
	  lengthDir=1;
	  if(whichLine-1 >= 0 && F < bufferIn[i+(whichLine-1)*W]){
	    F = bufferIn[i+(whichLine-1)*W];
	    indx[i] = i+(whichLine-1)*W; 
	    Dir = -1;
	    lengthDir=SQRT_2;
	  }
	  if(whichLine+1 < H && F < bufferIn[i+(whichLine+1)*W]){
	    indx[i] = i+(whichLine+1)*W; 
	    Dir = 1;
	    lengthDir=SQRT_2;
	  }
	  whichLine += Dir;
	}
	rank_open (bufferIn, indx, W, size, tolerance, bufferOut);
      }
			

      //Third direction Top to bottom
      //printf ("direction 3: Top to bottom\n");
      for(i=0;i<W;i+=step){
	wp=0;
	Length=0;
	stackSize=0;
	indx[0] = i;
	whichLine = i;
	for(j=1;j<H;j++){
	  F = bufferIn[whichLine+j*W];
	  indx[j] = whichLine+j*W; 
	  Dir = 0;
	  lengthDir=1;
	  if(whichLine-1 >= 0 && F < bufferIn[whichLine-1+ j*W]){
	    F = bufferIn[whichLine-1+ j*W];
	    indx[j] = whichLine-1+ j*W; 
	    Dir = -1;
	    lengthDir=SQRT_2;
	  }
	  if(whichLine+1 < W && F < bufferIn[whichLine+1+ j*W]){
	    indx[j] = whichLine+1+ j*W; 
	    Dir = 1;
	    lengthDir=SQRT_2;
	  }
	  whichLine += Dir;
	}
	rank_open (bufferIn, indx, H, size, tolerance, bufferOut);
      }
			
			
      //Fourth direction bottom to top
      //printf ("direction 4: bottom to top\n");
      for(i=0;i<W;i+=step){
	wp=0;
	Length=0;
	stackSize=0;
	indx[H-1] = i+(H-1)*W; 
	whichLine = i;
	for(j=H-2;j>=0;j--){
	  F = bufferIn[whichLine+j*W];
	  indx[j] = whichLine+j*W;
	  Dir = 0;
	  lengthDir=1;
	  if(whichLine-1 >= 0 && F < bufferIn[whichLine-1+ j*W]){
	    F = bufferIn[whichLine-1+ j*W];
	    indx[j] = whichLine-1+ j*W;
	    Dir = -1;
	    lengthDir=SQRT_2;
	  } 
	  if(whichLine+1 < W && F < bufferIn[whichLine+1+ j*W]){
	    indx[j] = whichLine+1+ j*W;
	    Dir = 1;
	    lengthDir=SQRT_2;
	  }
	  whichLine += Dir;
	}
	rank_open (bufferIn, indx, H, size, tolerance, bufferOut);
      }
      



      //Fifth direction bottom left, top right
      //printf ("direction 5: bottom left, top right\n");
      for(j=1;j<H;j+=step){
	wp=0;
	Length=0;
	stackSize=0;
	//BuildOpeningFromPoint_maxOp(bufferIn[j*W],j*W,MyStack,&stackSize,bufferOut,1);
	indx[0] = j*W;
	whichLine = j;
	i = 0;
	int cnt = 1;
				
	do
	  {
	    F = bufferIn[i+1 + (whichLine-1)*W];
	    indx[cnt] = i+1 + (whichLine-1)*W;
	    DirV = -1;
	    DirH = 1;
	    lengthDir=SQRT_2;
	    if(whichLine-1 >= 0 && F < bufferIn[i + (whichLine-1)*W]){
	      F = bufferIn[i + (whichLine-1)*W];
	      indx[cnt] = i + (whichLine-1)*W;
	      DirH = 0;
	      DirV = -1;
	      lengthDir=1;
	    }
	    if(i+1 <W && F < bufferIn[i+1 + (whichLine)*W]){
	      F = bufferIn[i+1 + (whichLine)*W];
	      indx[cnt] = i+1 + (whichLine)*W;
	      DirH = 1;
	      DirV = 0;
	      lengthDir=1;
	    }
	    i+=DirH;
	    whichLine+=DirV;
	    cnt++;
	    //BuildOpeningFromPoint_maxOp(bufferIn[i+ whichLine*W],i+ whichLine*W,MyStack,&stackSize,bufferOut,(float)lengthDir);
	  }
	while(i<=W-2 && whichLine>=1);
	//EndProcess_maxOp(MyStack,&stackSize,bufferOut);
	//printf ("cnt=%d\n", cnt);
	rank_open (bufferIn, indx, cnt, size, tolerance, bufferOut);
      }
      for(i=0;i<W-1;i+=step){
	wp=0;
	Length=0;
	stackSize=0;
	//BuildOpeningFromPoint_maxOp(bufferIn[i+(H-1)*W],i+(H-1)*W,MyStack,&stackSize,bufferOut,1);
	indx[0] = i+(H-1)*W;
	whichLine = i;
	j = H-1;
	int cnt = 1;
				
	do
	  {
	    F = bufferIn[whichLine+1 + (j-1)*W];
	    indx[cnt] = whichLine+1 + (j-1)*W;
	    DirV = -1;
	    DirH = 1;
	    lengthDir=SQRT_2;
	    if(j-1 >= 0 && F < bufferIn[whichLine + (j-1)*W]){
	      F = bufferIn[whichLine + (j-1)*W];
	      indx[cnt] = whichLine + (j-1)*W;
	      DirH = 0;
	      DirV = -1;
	      lengthDir=1;
	    }
	    if(whichLine+1 <W && F < bufferIn[whichLine+1 + (j)*W]){
	      F = bufferIn[whichLine+1 + (j)*W];
	      indx[cnt] = whichLine+1 + (j)*W;
	      DirH = 1;
	      DirV = 0;
	      lengthDir=1;
	    }
	    whichLine+=DirH;
	    j+=DirV;
	    cnt++;
	    //BuildOpeningFromPoint_maxOp(bufferIn[whichLine+ j*W],whichLine+ j*W,MyStack,&stackSize,bufferOut,(float)lengthDir);
	  }
	while(whichLine<=W-2 && j>=1);
	//printf ("cnt=%d\n", cnt);
	rank_open (bufferIn, indx, cnt, size, tolerance, bufferOut);
      }
			
      

      //Sixth direction top right to bottom left
      //printf ("direction 6: top right to bottom left\n");
      for(j=0;j<H-1;j+=step){
	wp=0;
	Length=0;
	stackSize=0;
	//BuildOpeningFromPoint_maxOp(bufferIn[W-1+j*W],W-1+j*W,MyStack,&stackSize,bufferOut,1);
	indx[0] = W-1+j*W;
	whichLine = j;
	i = W-1;
	int cnt = 1;			
	do
	  {
	    F = bufferIn[i-1 + (whichLine+1)*W];
	    indx[cnt] = i-1 + (whichLine+1)*W;
	    DirV = 1;
	    DirH = -1;
	    lengthDir=SQRT_2;
	    if(whichLine+1 < H && F < bufferIn[i + (whichLine+1)*W]){
	      F = bufferIn[i + (whichLine+1)*W];
	      indx[cnt] =  i + (whichLine+1)*W;
	      DirH = 0;
	      DirV = 1;
	      lengthDir=1;
	    }
	    if(i-1>=0 && F < bufferIn[i-1 + (whichLine)*W]){
	      F = bufferIn[i-1 + (whichLine)*W];
	      indx[cnt] = i-1 + (whichLine)*W;
	      DirH = -1;
	      DirV = 0;
	      lengthDir=1;
	    }
	    i+=DirH;
	    whichLine+=DirV;
	    cnt++; 
	    //BuildOpeningFromPoint_maxOp(bufferIn[i+ whichLine*W],i+ whichLine*W,MyStack,&stackSize,bufferOut,(float)lengthDir);
	  }
	while(i>=1 && whichLine<=H-2);
	//EndProcess_maxOp(MyStack,&stackSize,bufferOut);
	rank_open (bufferIn, indx, cnt, size, tolerance, bufferOut);
	}
      
      for(i=1;i<W;i+=step){
	wp=0;
	Length=0;
	stackSize=0;
	//BuildOpeningFromPoint_maxOp(bufferIn[i],i,MyStack,&stackSize,bufferOut,1);
	indx[0] = i;
	whichLine = i;
	j = 0;
	int cnt = 1;
				
	do
	  {
	    F = bufferIn[whichLine-1 + (j+1)*W];
	    indx[cnt] = whichLine-1 + (j+1)*W;
	    DirV = 1;
	    DirH = -1;
	    lengthDir=SQRT_2;
	    if(j+1 <H && F < bufferIn[whichLine + (j+1)*W]){
	      F = bufferIn[whichLine + (j+1)*W];
	      indx[cnt] = whichLine + (j+1)*W;
	      DirH = 0;
	      DirV = 1;
	      lengthDir=1;
	    }
	    if(whichLine-1 >=0 && F < bufferIn[whichLine-1 + (j)*W]){
	      F = bufferIn[whichLine-1 + (j)*W];
	      indx[cnt] = whichLine-1 + (j)*W; 
	      DirH = -1;
	      DirV = 0;
	      lengthDir=1;
	    }
	    whichLine+=DirH;
	    j+=DirV;
	    cnt++;
	    //BuildOpeningFromPoint_maxOp(bufferIn[whichLine+ j*W],whichLine+ j*W,MyStack,&stackSize,bufferOut,(float)lengthDir);
	  }
	while(whichLine>=1 && j<=H-2);
	//EndProcess_maxOp(MyStack,&stackSize,bufferOut);
	rank_open (bufferIn, indx, cnt, size, tolerance, bufferOut);
      }
     

      
      //Seventh direction top left, bottom right
      //printf ("direction 7: top left, bottom right\n");
      for(j=0;j<H-1;j+=step){
	wp=0;
	Length=0;
	stackSize=0;
	//BuildOpeningFromPoint_maxOp(bufferIn[j*W],j*W,MyStack,&stackSize,bufferOut,1);
	indx[0] = j*W;
	whichLine = j;
	i = 0;
	int cnt = 1;
				
	do
	  {
	    F = bufferIn[i+1 + (whichLine+1)*W];
	    indx[cnt] = i+1 + (whichLine+1)*W;
	    DirV = 1;
	    DirH = 1;
	    lengthDir=SQRT_2;
	    if(whichLine+1 <H && F < bufferIn[i + (whichLine+1)*W]){
	      F = bufferIn[i + (whichLine+1)*W];
	      indx[cnt] = i + (whichLine+1)*W;
	      DirH = 0;
	      DirV = 1;
	      lengthDir=1;
	    }
	    if(i+1 <W && F < bufferIn[i+1 + (whichLine)*W]){
	      F = bufferIn[i+1 + (whichLine)*W];
	      indx[cnt] = i+1 + (whichLine)*W;
	      DirH = 1;
	      DirV = 0;
	      lengthDir=1;
	    }
	    i+=DirH;
	    whichLine+=DirV;
	    cnt++;
	    // BuildOpeningFromPoint_maxOp(bufferIn[i+ whichLine*W],i+ whichLine*W,MyStack,&stackSize,bufferOut,(float)lengthDir);
	  }
	while(i<=W-2 && whichLine<=H-2);
	//EndProcess_maxOp(MyStack,&stackSize,bufferOut);
	rank_open (bufferIn, indx, cnt, size, tolerance, bufferOut);
      }
		
      for(i=0;i<W-1;i+=step){
	wp=0;
	Length=0;
	stackSize=0;
	//BuildOpeningFromPoint_maxOp(bufferIn[i],i,MyStack,&stackSize,bufferOut,1);
	indx[0] = i;
	whichLine = i;
	j = 0;
	int cnt=1;
				
	do
	  {
	    F = bufferIn[whichLine+1 + (j+1)*W];
	    indx[cnt] = whichLine+1 + (j+1)*W; 
	    DirV = 1;
	    DirH = 1;
	    lengthDir=SQRT_2;
	    if(j+1 < H && F < bufferIn[whichLine + (j+1)*W]){
	      F = bufferIn[whichLine + (j+1)*W];
	      indx[cnt] = whichLine + (j+1)*W;
	      DirH = 0;
	      DirV = 1;
	      lengthDir=1;
	    }
	    if(whichLine+1 <W && F < bufferIn[whichLine+1 + (j)*W]){
	      F = bufferIn[whichLine+1 + (j)*W];
	      indx[cnt] = whichLine+1 + (j)*W;
	      DirH = 1;
	      DirV = 0;
	      lengthDir=1;
	    }
	    whichLine+=DirH;
	    j+=DirV;
	    cnt++;
	    //BuildOpeningFromPoint_maxOp(bufferIn[whichLine+ j*W],whichLine+ j*W,MyStack,&stackSize,bufferOut,(float)lengthDir);
	  }
	while(whichLine<=W-2 && j<=H-2);
	//EndProcess_maxOp(MyStack,&stackSize,bufferOut);
	rank_open (bufferIn, indx, cnt, size, tolerance, bufferOut);
      }



      //8 : direction bottom right to top left
      //printf ("direction 8: bottom right to top left\n");
      for(j=1;j<H;j+=step){
	wp=0;
	Length=0;
	stackSize=0;
	//BuildOpeningFromPoint_maxOp(bufferIn[W-1+j*W],W-1+j*W,MyStack,&stackSize,bufferOut,1);
	indx[0] = W-1+j*W;
	whichLine = j;
	i = W-1;
	int cnt = 1;

	do
	  {
	    F = bufferIn[i-1 + (whichLine-1)*W];
	    indx[cnt] = i-1 + (whichLine-1)*W;
	    DirV = -1;
	    DirH = -1;
	    lengthDir=SQRT_2;
	    if(whichLine-1 >=0 && F < bufferIn[i + (whichLine-1)*W]){
	      F = bufferIn[i + (whichLine-1)*W];
	      indx[cnt] = i + (whichLine-1)*W; 
	      DirH = 0;
	      DirV = -1;
	      lengthDir=1;
	    }
	    if(i-1>=0 && F < bufferIn[i-1 + (whichLine)*W]){
	      F = bufferIn[i-1 + (whichLine)*W];
	      indx[cnt] = i-1 + (whichLine)*W; 
	      DirH = -1;
	      DirV = 0;
	      lengthDir=1;
	    }
	    i+=DirH;
	    whichLine+=DirV;
	    cnt++; 
	    //BuildOpeningFromPoint_maxOp(bufferIn[i+ whichLine*W],i+ whichLine*W,MyStack,&stackSize,bufferOut,(float)lengthDir);
	  }
	while(i>=1 && whichLine>=1);
	// EndProcess_maxOp(MyStack,&stackSize,bufferOut);
	rank_open (bufferIn, indx, cnt, size, tolerance, bufferOut);
      }
      for(i=1;i<W;i+=step){
	wp=0;
	Length=0;
	stackSize=0;
	//BuildOpeningFromPoint_maxOp(bufferIn[i+(H-1)*W],i+(H-1)*W,MyStack,&stackSize,bufferOut,1);
	indx[0] = i+(H-1)*W; 
	whichLine = i;
	j = H-1;
	int cnt = 1;
				
	do
	  {
	    F = bufferIn[whichLine-1 + (j-1)*W];
	    indx[cnt] = whichLine-1 + (j-1)*W; 
	    DirV = -1;
	    DirH = -1;
	    lengthDir=SQRT_2;
	    if(j-1 >=0 && F < bufferIn[whichLine + (j-1)*W]){
	      F = bufferIn[whichLine + (j-1)*W];
	      indx[cnt] = whichLine + (j-1)*W; 
	      DirH = 0;
	      DirV = -1;
	      lengthDir=1;
	    }
	    if(whichLine-1 >=0 && F < bufferIn[whichLine-1 + (j)*W]){
	      F = bufferIn[whichLine-1 + (j)*W];
	      indx[cnt] = whichLine-1 + (j)*W; 
	      DirH = -1;
	      DirV = 0;
	      lengthDir=1;
	    }
	    whichLine+=DirH;
	    j+=DirV;
	    cnt++; 
	    //BuildOpeningFromPoint_maxOp(bufferIn[whichLine+ j*W],whichLine+ j*W,MyStack,&stackSize,bufferOut,(float)lengthDir);
	  }
	while(whichLine>=1 && j>=1);
	//EndProcess_maxOp(MyStack,&stackSize,bufferOut);
	rank_open (bufferIn, indx, cnt, size, tolerance, bufferOut);
      }

			
      delete []LineIdx;
      delete []indx;

      if(rec){	//Reconstruction
	RES_C res = t_ImUnderBuild(imIn,imOut,imOut);
	if(res !=RES_OK){
	  MORPHEE_REGISTER_ERROR("Error in t_ImParsimoniousPathOpening in function t_ImUnderBuild");
	  return res;
	}
      }
      return RES_OK;
    }

  }
}//morphee


#endif
			
