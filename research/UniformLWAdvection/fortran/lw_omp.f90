! Constant-carrier, periodic cell-average Lax-Wendroff reference for the Julia
! UniformLWAdvection toy. Each directional sweep recomputes both oriented face
! fluxes, and the observable q field is updated after every complete step.
module uniform_lw_openmp
  use, intrinsic :: iso_fortran_env, only: real64
  use omp_lib
  implicit none
  private
  public :: sweep_x, sweep_y

contains

  pure elemental real(real64) function lwflux(left, right, speed, ratio) result(flux)
    real(real64), intent(in) :: left, right, speed, ratio
    flux = speed * ((left + right) * 0.5_real64) - &
      ((speed * speed) * ratio * 0.5_real64) * (right - left)
  end function lwflux

  subroutine x_tile(nx, ny, out, q, u, ratio, ilo, ihi, jlo, jhi)
    integer, intent(in) :: nx, ny, ilo, ihi, jlo, jhi
    real(real64), intent(in) :: q(nx,ny), u(nx,ny), ratio
    real(real64), intent(inout) :: out(nx,ny)
    integer :: i, j
    real(real64) :: qi, fr, fl
    do j = jlo, jhi
      if (ilo == 1) then
        qi = q(1,j)
        fr = lwflux(qi, q(min(2,nx),j), u(1,j), ratio)
        fl = lwflux(q(nx,j), qi, u(nx,j), ratio)
        out(1,j) = qi - ratio * (fr - fl)
      end if
      !$omp simd private(qi,fr,fl)
      do i = max(2,ilo), min(nx-1,ihi)
        qi = q(i,j)
        fr = lwflux(qi, q(i+1,j), u(i,j), ratio)
        fl = lwflux(q(i-1,j), qi, u(i-1,j), ratio)
        out(i,j) = qi - ratio * (fr - fl)
      end do
      if (ihi == nx .and. nx > 1) then
        qi = q(nx,j)
        fr = lwflux(qi, q(1,j), u(nx,j), ratio)
        fl = lwflux(q(nx-1,j), qi, u(nx-1,j), ratio)
        out(nx,j) = qi - ratio * (fr - fl)
      end if
    end do
  end subroutine x_tile

  subroutine y_tile(nx, ny, out, q, v, ratio, ilo, ihi, jlo, jhi)
    integer, intent(in) :: nx, ny, ilo, ihi, jlo, jhi
    real(real64), intent(in) :: q(nx,ny), v(nx,ny), ratio
    real(real64), intent(inout) :: out(nx,ny)
    integer :: i, j, jm, jp
    real(real64) :: qi, ft, fb
    do j = jlo, jhi
      jm = j-1
      jp = j+1
      if (j == 1) jm = ny
      if (j == ny) jp = 1
      !$omp simd private(qi,ft,fb)
      do i = ilo, ihi
        qi = q(i,j)
        ft = lwflux(qi, q(i,jp), v(i,j), ratio)
        fb = lwflux(q(i,jm), qi, v(i,jm), ratio)
        out(i,j) = qi - ratio * (ft - fb)
      end do
    end do
  end subroutine y_tile

  subroutine sweep_x(nx,ny,out,q,u,ratio,workers,tile_x,tile_y,track,used)
    integer, intent(in) :: nx,ny,workers,tile_x,tile_y
    real(real64), intent(out) :: out(nx,ny)
    real(real64), intent(in) :: q(nx,ny),u(nx,ny),ratio
    logical, intent(in) :: track
    integer, intent(inout) :: used(workers)
    integer :: t, nt, ntx, ilo, ihi, jlo, jhi, tid
    ntx = (nx+tile_x-1)/tile_x
    nt = ntx*((ny+tile_y-1)/tile_y)
    if (workers == 1) then
      if (.not. track) then
        call x_tile(nx,ny,out,q,u,ratio,1,nx,1,ny)
        return
      end if
      if (track) used(1) = 1
      do t=1,nt
        ilo = mod(t-1,ntx)*tile_x+1
        ihi = min(nx,ilo+tile_x-1)
        jlo = ((t-1)/ntx)*tile_y+1
        jhi = min(ny,jlo+tile_y-1)
        call x_tile(nx,ny,out,q,u,ratio,ilo,ihi,jlo,jhi)
      end do
      return
    end if
    !$omp parallel do default(none) schedule(static) num_threads(workers) &
    !$omp shared(nx,ny,out,q,u,ratio,workers,tile_x,tile_y,track,used,nt,ntx) &
    !$omp private(t,ilo,ihi,jlo,jhi,tid)
    do t = 1, nt
      ilo = mod(t-1,ntx)*tile_x+1
      ihi = min(nx,ilo+tile_x-1)
      jlo = ((t-1)/ntx)*tile_y+1
      jhi = min(ny,jlo+tile_y-1)
      if (track) then
        tid = omp_get_thread_num()+1
        used(tid) = 1
      end if
      call x_tile(nx,ny,out,q,u,ratio,ilo,ihi,jlo,jhi)
    end do
    !$omp end parallel do
  end subroutine sweep_x

  subroutine sweep_y(nx,ny,out,q,v,ratio,workers,tile_x,tile_y,track,used)
    integer, intent(in) :: nx,ny,workers,tile_x,tile_y
    real(real64), intent(out) :: out(nx,ny)
    real(real64), intent(in) :: q(nx,ny),v(nx,ny),ratio
    logical, intent(in) :: track
    integer, intent(inout) :: used(workers)
    integer :: t, nt, ntx, ilo, ihi, jlo, jhi, tid
    ntx = (nx+tile_x-1)/tile_x
    nt = ntx*((ny+tile_y-1)/tile_y)
    if (workers == 1) then
      if (.not. track) then
        call y_tile(nx,ny,out,q,v,ratio,1,nx,1,ny)
        return
      end if
      if (track) used(1) = 1
      do t=1,nt
        ilo = mod(t-1,ntx)*tile_x+1
        ihi = min(nx,ilo+tile_x-1)
        jlo = ((t-1)/ntx)*tile_y+1
        jhi = min(ny,jlo+tile_y-1)
        call y_tile(nx,ny,out,q,v,ratio,ilo,ihi,jlo,jhi)
      end do
      return
    end if
    !$omp parallel do default(none) schedule(static) num_threads(workers) &
    !$omp shared(nx,ny,out,q,v,ratio,workers,tile_x,tile_y,track,used,nt,ntx) &
    !$omp private(t,ilo,ihi,jlo,jhi,tid)
    do t = 1, nt
      ilo = mod(t-1,ntx)*tile_x+1
      ihi = min(nx,ilo+tile_x-1)
      jlo = ((t-1)/ntx)*tile_y+1
      jhi = min(ny,jlo+tile_y-1)
      if (track) then
        tid = omp_get_thread_num()+1
        used(tid) = 1
      end if
      call y_tile(nx,ny,out,q,v,ratio,ilo,ihi,jlo,jhi)
    end do
    !$omp end parallel do
  end subroutine sweep_y

end module uniform_lw_openmp

program lw_omp
  use, intrinsic :: iso_fortran_env, only: real64, error_unit, output_unit, compiler_version
  use, intrinsic :: iso_c_binding, only: c_int, c_int8_t, c_size_t
  use omp_lib
  use uniform_lw_openmp
  implicit none
  interface
    function sched_setaffinity(pid, nbytes, mask) bind(c, name='sched_setaffinity') result(rc)
      import :: c_int, c_size_t, c_int8_t
      integer(c_int), value :: pid
      integer(c_size_t), value :: nbytes
      integer(c_int8_t), intent(in) :: mask(*)
      integer(c_int) :: rc
    end function
    function sched_getaffinity(pid, nbytes, mask) bind(c, name='sched_getaffinity') result(rc)
      import :: c_int, c_size_t, c_int8_t
      integer(c_int), value :: pid
      integer(c_size_t), value :: nbytes
      integer(c_int8_t), intent(out) :: mask(*)
      integer(c_int) :: rc
    end function
    function sched_getcpu() bind(c, name='sched_getcpu') result(cpu)
      import :: c_int
      integer(c_int) :: cpu
    end function
  end interface
  integer :: nx,ny,steps,workers,tile_x,tile_y,argc,k,ios,unit,pass
  integer :: counts(3)
  integer, allocatable :: used(:),probed(:,:),cpus(:),observed_cpus(:),pin_ok(:)
  real(real64), allocatable :: q(:,:),q0(:,:),probe_ref(:,:),u(:,:),v(:,:),first(:,:),second(:,:)
  real(real64) :: uspeed,vspeed,dt,time0,elapsed,final_time,mass,checksum,pi,x,y
  character(len=1024) :: key,value,input_path,output_path,cpus_text
  logical :: have_dt

  nx=512; ny=512; steps=1; workers=1; tile_x=512; tile_y=16
  uspeed=0.7_real64; vspeed=-0.3_real64; dt=0.0_real64
  input_path=''; output_path=''; cpus_text=''; have_dt=.false.
  argc=command_argument_count()
  if (mod(argc,2) /= 0) call fail('expected --key value argument pairs')
  do k=1,argc,2
    call get_command_argument(k,key)
    call get_command_argument(k+1,value)
    select case(trim(key))
    case('--nx')
      read(value,*,iostat=ios) nx
    case('--ny')
      read(value,*,iostat=ios) ny
    case('--steps')
      read(value,*,iostat=ios) steps
    case('--threads')
      read(value,*,iostat=ios) workers
    case('--tile-x')
      read(value,*,iostat=ios) tile_x
    case('--tile-y')
      read(value,*,iostat=ios) tile_y
    case('--u')
      read(value,*,iostat=ios) uspeed
    case('--v')
      read(value,*,iostat=ios) vspeed
    case('--dt')
      read(value,*,iostat=ios) dt
      have_dt=.true.
    case('--input')
      input_path=trim(value)
      ios=0
    case('--output')
      output_path=trim(value)
      ios=0
    case('--cpus')
      cpus_text=trim(value)
      ios=0
    case default
      call fail('unrecognized argument: '//trim(key))
    end select
    if (ios /= 0) call fail('invalid value for '//trim(key))
  end do
  if (nx<1 .or. ny<1 .or. steps<1 .or. workers<1 .or. tile_x<1 .or. tile_y<1) &
    call fail('dimensions, steps, threads and tile sizes must be positive')
  if (.not. have_dt) dt=0.35_real64/(0.7_real64*nx+0.3_real64*ny)
  if (.not. (dt>=0.0_real64 .and. abs(uspeed)*dt*nx*0.5_real64<=1.0_real64 &
      .and. abs(vspeed)*dt*ny<=1.0_real64)) call fail('unstable or invalid dt')
  call omp_set_dynamic(.false.)
  call omp_set_num_threads(workers)
  allocate(cpus(workers),observed_cpus(workers),pin_ok(workers))
  cpus=-1; observed_cpus=-1; pin_ok=0
  if (len_trim(cpus_text)>0) then
    call parse_cpu_list(trim(cpus_text),cpus)
    call bind_workers()
    if (any(pin_ok/=1)) call fail('worker CPU binding or affinity verification failed')
  else
    call observe_workers()
  end if

  allocate(q(nx,ny),q0(nx,ny),probe_ref(nx,ny),u(nx,ny),v(nx,ny),first(nx,ny),second(nx,ny))
  allocate(used(workers),probed(workers,3))
  u=uspeed; v=vspeed
  if (len_trim(input_path)>0) then
    open(newunit=unit,file=trim(input_path),access='stream',form='unformatted',status='old',action='read',iostat=ios)
    if (ios/=0) call fail('cannot open input field')
    read(unit,iostat=ios) q0
    close(unit)
    if (ios/=0) call fail('cannot read input field')
  else
    ! The benchmark harness normally supplies Julia's exact initial cell
    ! averages. This independent fixture is useful for stand-alone runs.
    pi=acos(-1.0_real64)
    do pass=1,ny
      y=(real(pass,real64)-0.5_real64)/ny
      do k=1,nx
        x=(real(k,real64)-0.5_real64)/nx
        q0(k,pass)=1.0_real64 + 0.2_real64*sinc(1.0_real64/nx)*sin(2*pi*x) &
          + 0.1_real64*sinc(2.0_real64/ny)*cos(4*pi*y) &
          + 0.15_real64*sinc(1.0_real64/nx)*sinc(1.0_real64/ny)*sin(2*pi*(x+y))
      end do
    end do
  end if
  q=q0
  call step(.false.)
  call step(.false.)
  q=q0
  call step(.false.)
  probe_ref=q
  q=q0
  used=0; call sweep_x(nx,ny,first,q,u,dt*nx*0.5_real64,workers,tile_x,tile_y,.true.,used)
  probed(:,1)=used
  counts(1)=sum(used)
  used=0; call sweep_y(nx,ny,second,first,v,dt*ny,workers,tile_x,tile_y,.true.,used)
  probed(:,2)=used
  counts(2)=sum(used)
  used=0; call sweep_x(nx,ny,q,second,u,dt*nx*0.5_real64,workers,tile_x,tile_y,.true.,used)
  probed(:,3)=used
  counts(3)=sum(used)
  if (maxval(abs(q-probe_ref))>0.0_real64) call fail('tracked worker probe changed the field')
  deallocate(probe_ref)
  q=q0
  final_time=0.0_real64
  time0=omp_get_wtime()
  do k=1,steps
    call step(.false.)
    final_time=final_time+dt
  end do
  elapsed=omp_get_wtime()-time0
  checksum=sum(q)
  mass=checksum/real(nx,real64)/real(ny,real64)

  if (len_trim(output_path)>0) then
    open(newunit=unit,file=trim(output_path),access='stream',form='unformatted',status='replace',action='write',iostat=ios)
    if (ios/=0) call fail('cannot open output field')
    write(unit,iostat=ios) q
    close(unit)
    if (ios/=0) call fail('cannot write output field')
  end if
  write(output_unit,'(a)',advance='no') '{"implementation":"fortran_openmp",'
  write(output_unit,'(a,i0,a,i0,a,i0,a,i0,a,i0,a,i0,a)',advance='no') &
    '"nx":',nx,',"ny":',ny,',"steps":',steps,',"workers":',workers, &
    ',"tile_x":',tile_x,',"tile_y":',tile_y,','
  write(output_unit,'(a,es24.16e3,a,es24.16e3,a,es24.16e3,a)',advance='no') &
    '"dt":',dt,',"time_seconds":',elapsed, &
    ',"seconds_per_unknown_step":',elapsed/(real(nx,real64)*real(ny,real64)*real(steps,real64)),','
  write(output_unit,'(a,es24.16e3,a,es24.16e3,a,es24.16e3,a)',advance='no') &
    '"final_time":',final_time,',"mass":',mass,',"checksum":',checksum,','
  write(output_unit,'(a,i0,a,i0,a,i0,a)',advance='no') &
    '"active_worker_counts":[',counts(1),',',counts(2),',',counts(3),'],'
  write(output_unit,'(a)',advance='no') '"active_worker_ids_by_pass":['
  do pass=1,3
    if (pass>1) write(output_unit,'(a)',advance='no') ','
    write(output_unit,'(a)',advance='no') '['
    ios=0
    do k=1,workers
      if (probed(k,pass)==0) cycle
      if (ios>0) write(output_unit,'(a)',advance='no') ','
      write(output_unit,'(i0)',advance='no') k
      ios=ios+1
    end do
    write(output_unit,'(a)',advance='no') ']'
  end do
  write(output_unit,'(a)',advance='no') '],"affinity_cpu_ids":['
  do k=1,workers
    if (k>1) write(output_unit,'(a)',advance='no') ','
    write(output_unit,'(i0)',advance='no') cpus(k)
  end do
  write(output_unit,'(a)',advance='no') '],"observed_cpu_ids":['
  do k=1,workers
    if (k>1) write(output_unit,'(a)',advance='no') ','
    write(output_unit,'(i0)',advance='no') observed_cpus(k)
  end do
  write(output_unit,'(a)',advance='no') '],"pin_verified":'
  if (len_trim(cpus_text)>0) then
    write(output_unit,'(a)',advance='no') 'true'
  else
    write(output_unit,'(a)',advance='no') 'false'
  end if
  write(output_unit,'(a)') ',"compiler":"'//trim(compiler_version())//'"}'

contains

  subroutine step(track)
    logical, intent(in) :: track
    if (track) used=0
    call sweep_x(nx,ny,first,q,u,dt*nx*0.5_real64,workers,tile_x,tile_y,track,used)
    call sweep_y(nx,ny,second,first,v,dt*ny,workers,tile_x,tile_y,track,used)
    call sweep_x(nx,ny,q,second,u,dt*nx*0.5_real64,workers,tile_x,tile_y,track,used)
  end subroutine step

  subroutine parse_cpu_list(list,ids)
    character(len=*), intent(in) :: list
    integer, intent(out) :: ids(:)
    integer :: n,start,stop,sep,stat
    start=1
    do n=1,size(ids)
      sep=index(list(start:),',')
      if (sep==0) then
        stop=len_trim(list)
        if (n/=size(ids)) call fail('CPU list has fewer entries than workers')
      else
        stop=start+sep-2
        if (n==size(ids)) call fail('CPU list has more entries than workers')
      end if
      read(list(start:stop),*,iostat=stat) ids(n)
      if (stat/=0 .or. ids(n)<0 .or. ids(n)>=1024) call fail('invalid CPU ID')
      start=stop+2
    end do
    do n=1,size(ids)
      if (count(ids==ids(n))/=1) call fail('CPU list contains duplicate IDs')
    end do
  end subroutine parse_cpu_list

  subroutine bind_workers()
    integer :: id,cpu,rc
    integer(c_int8_t) :: mask(128),actual(128)
    !$omp parallel default(none) num_threads(workers) &
    !$omp shared(workers,cpus,observed_cpus,pin_ok) private(id,cpu,rc,mask,actual)
    id=omp_get_thread_num()+1
    cpu=cpus(id)
    mask=0_c_int8_t
    mask(cpu/8+1)=ibset(mask(cpu/8+1),mod(cpu,8))
    rc=sched_setaffinity(0_c_int,128_c_size_t,mask)
    if (rc==0) then
      rc=sched_getaffinity(0_c_int,128_c_size_t,actual)
      if (rc==0 .and. all(actual==mask)) pin_ok(id)=1
    end if
    observed_cpus(id)=sched_getcpu()
    !$omp end parallel
  end subroutine bind_workers

  subroutine observe_workers()
    integer :: id
    !$omp parallel default(none) num_threads(workers) &
    !$omp shared(observed_cpus) private(id)
    id=omp_get_thread_num()+1
    observed_cpus(id)=sched_getcpu()
    !$omp end parallel
  end subroutine observe_workers

  pure real(real64) function sinc(z)
    real(real64), intent(in) :: z
    sinc=sin(acos(-1.0_real64)*z)/(acos(-1.0_real64)*z)
  end function sinc

  subroutine fail(message)
    character(len=*), intent(in) :: message
    write(error_unit,'(a)') trim(message)
    stop 1
  end subroutine fail
end program lw_omp
