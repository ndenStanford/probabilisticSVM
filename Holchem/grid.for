      program Nmcsgsim2
c ******************************************************************

c ******************************************************************

      
      real*8 yy,x,y
      integer N,M
	
c      
      N=100 ! (number of  colums)		
	M=100 !(rows)
	
	DX=1.0	! Grid
	Dy=1.0	!

      open(unit=11,file="lnk.dat")   !write grid name
	read(11,*)	 ! skipt no data lines 
	read(11,*)
	read(11,*)
      
      
	open(unit=12,file="lnk.out")
	write(12,*) N, M
	write (12,*) "campo di Y"
	write (12,*) "x, y, Y" 
	



	DO i=1,M

        do j=1,N
          ind=ind+1
		 x = dx/2 + (i-1)*dx
	     y = dy/2 + (j-1)*dy
	    
		 read(11,*)yy
	     write(12,*)x, y, yy
		

      end do    
     
      end do

      
	close(11)
	close(12)	
       
	
     
     

       
            
      stop
      end
