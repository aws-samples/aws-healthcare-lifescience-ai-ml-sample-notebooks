/*
==============================================================================
   US-align: universal structure alignment of monomeric and complex proteins
   and nucleic acids

   This program was written by Chengxin Zhang at Yang Zhang lab,
   Department of Computational Medicine and Bioinformatics,
   University of Michigan, 100 Washtenaw Ave, Ann Arbor, MI 48109-2218.
   Please report issues to yangzhanglab@umich.edu

   Reference:
   * Chengxin Zhang, Morgan Shine, Anna Marie Pyle, Yang Zhang
     (2022) Nat Methods
   * Chengxin Zhang, Anna Marie Pyle (2022) iScience

   DISCLAIMER:
     Permission to use, copy, modify, and distribute this program for 
     any purpose, with or without fee, is hereby granted, provided that
     the notices on the head, the reference information, and this
     copyright notice appear in all copies or substantial portions of 
     the Software. It is provided "as is" without express or implied 
     warranty.
===============================================================================

=========================
 How to install US-align
=========================
To compile the program in your Linux computer, simply enter

    make

or

    g++ -static -O3 -ffast-math -lm -o USalign USalign.cpp

The '-static' flag should be removed on Mac OS, which does not support
building static executables.

USalign compiled on Linux, Mac OS and Linux Subsystem for Windows (WSL2) on
Windows 10 onwards can read both uncompressed files and gz compressed
files, provided that the "gunzip" command is available. On the other hand, due
to the lack of POSIX support on Windows, US-align natively compiled on Windows
without WSL2 cannot parse gz compressed files.

=====================
 How to use US-align
=====================
You can run the program without arguments to obtain a brief instruction

    ./USalign structure1.pdb structure2.pdb

A full list of available options can be explored by:

    ./USalign -h
*/


#include <cfloat>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <string>
#include <iomanip>
#include <map>

using namespace std;

/* pstream.h */
/* the following code block for REDI_PSTREAM_H_SEEN is from Jonathan
 * Wakely's pstream.h. His original copyright notice is shown below */
// PStreams - POSIX Process I/O for C++

//        Copyright (C) 2001 - 2017 Jonathan Wakely
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//

/* do not compile on windows, which does not have cygwin */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) && !defined(__CYGWIN__)
#define NO_PSTREAM
#else

#ifndef REDI_PSTREAM_H_SEEN
#define REDI_PSTREAM_H_SEEN

#include <ios>
#include <streambuf>
#include <istream>
#include <ostream>
#include <string>
#include <vector>
#include <algorithm>    // for min()
#include <cerrno>       // for errno
#include <cstddef>      // for size_t, NULL
#include <cstdlib>      // for exit()
#include <sys/types.h>  // for pid_t
#include <sys/wait.h>   // for waitpid()
#include <sys/ioctl.h>  // for ioctl() and FIONREAD
#if defined(__sun)
# include <sys/filio.h> // for FIONREAD on Solaris 2.5
#endif
#include <unistd.h>     // for pipe() fork() exec() and filedes functions
#include <signal.h>     // for kill()
#include <fcntl.h>      // for fcntl()
#if REDI_EVISCERATE_PSTREAMS
# include <stdio.h>     // for FILE, fdopen()
#endif


/// The library version.
#define PSTREAMS_VERSION 0x0101   // 1.0.1

/**
 *  @namespace redi
 *  @brief  All PStreams classes are declared in namespace redi.
 *
 *  Like the standard iostreams, PStreams is a set of class templates,
 *  taking a character type and traits type. As with the standard streams
 *  they are most likely to be used with @c char and the default
 *  traits type, so typedefs for this most common case are provided.
 *
 *  The @c pstream_common class template is not intended to be used directly,
 *  it is used internally to provide the common functionality for the
 *  other stream classes.
 */
namespace redi
{
  /// Common base class providing constants and typenames.
  struct pstreams
  {
    /// Type used to specify how to connect to the process.
    typedef std::ios_base::openmode           pmode;

    /// Type used to hold the arguments for a command.
    typedef std::vector<std::string>          argv_type;

    /// Type used for file descriptors.
    typedef int                               fd_type;

    static const pmode pstdin  = std::ios_base::out; ///< Write to stdin
    static const pmode pstdout = std::ios_base::in;  ///< Read from stdout
    static const pmode pstderr = std::ios_base::app; ///< Read from stderr

    /// Create a new process group for the child process.
    static const pmode newpg   = std::ios_base::trunc;

  protected:
    enum { bufsz = 32 };  ///< Size of pstreambuf buffers.
    enum { pbsz  = 2 };   ///< Number of putback characters kept.
  };

  /// Class template for stream buffer.
  template <typename CharT, typename Traits = std::char_traits<CharT> >
    class basic_pstreambuf
    : public std::basic_streambuf<CharT, Traits>
    , public pstreams
    {
    public:
      // Type definitions for dependent types
      typedef CharT                             char_type;
      typedef Traits                            traits_type;
      typedef typename traits_type::int_type    int_type;
      typedef typename traits_type::off_type    off_type;
      typedef typename traits_type::pos_type    pos_type;
      /** @deprecated use pstreams::fd_type instead. */
      typedef fd_type                           fd_t;

      /// Default constructor.
      basic_pstreambuf();

      /// Constructor that initialises the buffer with @a cmd.
      basic_pstreambuf(const std::string& cmd, pmode mode);

      /// Constructor that initialises the buffer with @a file and @a argv.
      basic_pstreambuf( const std::string& file,
                        const argv_type& argv,
                        pmode mode );

      /// Destructor.
      ~basic_pstreambuf();

      /// Initialise the stream buffer with @a cmd.
      basic_pstreambuf*
      open(const std::string& cmd, pmode mode);

      /// Initialise the stream buffer with @a file and @a argv.
      basic_pstreambuf*
      open(const std::string& file, const argv_type& argv, pmode mode);

      /// Close the stream buffer and wait for the process to exit.
      basic_pstreambuf*
      close();

      /// Send a signal to the process.
      basic_pstreambuf*
      kill(int signal = SIGTERM);

      /// Send a signal to the process' process group.
      basic_pstreambuf*
      killpg(int signal = SIGTERM);

      /// Close the pipe connected to the process' stdin.
      void
      peof();

      /// Change active input source.
      bool
      read_err(bool readerr = true);

      /// Report whether the stream buffer has been initialised.
      bool
      is_open() const;

      /// Report whether the process has exited.
      bool
      exited();

#if REDI_EVISCERATE_PSTREAMS
      /// Obtain FILE pointers for each of the process' standard streams.
      std::size_t
      fopen(FILE*& in, FILE*& out, FILE*& err);
#endif

      /// Return the exit status of the process.
      int
      status() const;

      /// Return the error number (errno) for the most recent failed operation.
      int
      error() const;

    protected:
      /// Transfer characters to the pipe when character buffer overflows.
      int_type
      overflow(int_type c);

      /// Transfer characters from the pipe when the character buffer is empty.
      int_type
      underflow();

      /// Make a character available to be returned by the next extraction.
      int_type
      pbackfail(int_type c = traits_type::eof());

      /// Write any buffered characters to the stream.
      int
      sync();

      /// Insert multiple characters into the pipe.
      std::streamsize
      xsputn(const char_type* s, std::streamsize n);

      /// Insert a sequence of characters into the pipe.
      std::streamsize
      write(const char_type* s, std::streamsize n);

      /// Extract a sequence of characters from the pipe.
      std::streamsize
      read(char_type* s, std::streamsize n);

      /// Report how many characters can be read from active input without blocking.
      std::streamsize
      showmanyc();

    protected:
      /// Enumerated type to indicate whether stdout or stderr is to be read.
      enum buf_read_src { rsrc_out = 0, rsrc_err = 1 };

      /// Initialise pipes and fork process.
      pid_t
      fork(pmode mode);

      /// Wait for the child process to exit.
      int
      wait(bool nohang = false);

      /// Return the file descriptor for the output pipe.
      fd_type&
      wpipe();

      /// Return the file descriptor for the active input pipe.
      fd_type&
      rpipe();

      /// Return the file descriptor for the specified input pipe.
      fd_type&
      rpipe(buf_read_src which);

      void
      create_buffers(pmode mode);

      void
      destroy_buffers(pmode mode);

      /// Writes buffered characters to the process' stdin pipe.
      bool
      empty_buffer();

      bool
      fill_buffer(bool non_blocking = false);

      /// Return the active input buffer.
      char_type*
      rbuffer();

      buf_read_src
      switch_read_buffer(buf_read_src);

    private:
      basic_pstreambuf(const basic_pstreambuf&);
      basic_pstreambuf& operator=(const basic_pstreambuf&);

      void
      init_rbuffers();

      pid_t         ppid_;        // pid of process
      fd_type       wpipe_;       // pipe used to write to process' stdin
      fd_type       rpipe_[2];    // two pipes to read from, stdout and stderr
      char_type*    wbuffer_;
      char_type*    rbuffer_[2];
      char_type*    rbufstate_[3];
      /// Index into rpipe_[] to indicate active source for read operations.
      buf_read_src  rsrc_;
      int           status_;      // hold exit status of child process
      int           error_;       // hold errno if fork() or exec() fails
    };

  /// Class template for common base class.
  template <typename CharT, typename Traits = std::char_traits<CharT> >
    class pstream_common
    : virtual public std::basic_ios<CharT, Traits>
    , virtual public pstreams
    {
    protected:
      typedef basic_pstreambuf<CharT, Traits>       streambuf_type;

      typedef pstreams::pmode                       pmode;
      typedef pstreams::argv_type                   argv_type;

      /// Default constructor.
      pstream_common();

      /// Constructor that initialises the stream by starting a process.
      pstream_common(const std::string& cmd, pmode mode);

      /// Constructor that initialises the stream by starting a process.
      pstream_common(const std::string& file, const argv_type& argv, pmode mode);

      /// Pure virtual destructor.
      virtual
      ~pstream_common() = 0;

      /// Start a process.
      void
      do_open(const std::string& cmd, pmode mode);

      /// Start a process.
      void
      do_open(const std::string& file, const argv_type& argv, pmode mode);

    public:
      /// Close the pipe.
      void
      close();

      /// Report whether the stream's buffer has been initialised.
      bool
      is_open() const;

      /// Return the command used to initialise the stream.
      const std::string&
      command() const;

      /// Return a pointer to the stream buffer.
      streambuf_type*
      rdbuf() const;

#if REDI_EVISCERATE_PSTREAMS
      /// Obtain FILE pointers for each of the process' standard streams.
      std::size_t
      fopen(FILE*& in, FILE*& out, FILE*& err);
#endif

    protected:
      std::string       command_; ///< The command used to start the process.
      streambuf_type    buf_;     ///< The stream buffer.
    };


  /**
   * @class basic_ipstream
   * @brief Class template for Input PStreams.
   *
   * Reading from an ipstream reads the command's standard output and/or
   * standard error (depending on how the ipstream is opened)
   * and the command's standard input is the same as that of the process
   * that created the object, unless altered by the command itself.
   */

  template <typename CharT, typename Traits = std::char_traits<CharT> >
    class basic_ipstream
    : public std::basic_istream<CharT, Traits>
    , public pstream_common<CharT, Traits>
    , virtual public pstreams
    {
      typedef std::basic_istream<CharT, Traits>     istream_type;
      typedef pstream_common<CharT, Traits>         pbase_type;

      using pbase_type::buf_;  // declare name in this scope

      // Ensure a basic_ipstream will read from at least one pipe
      pmode readable(pmode mode)
      {
        if (!(mode & (pstdout|pstderr)))
          mode |= pstdout;
        return mode;
      }

    public:
      /// Type used to specify how to connect to the process.
      typedef typename pbase_type::pmode            pmode;

      /// Type used to hold the arguments for a command.
      typedef typename pbase_type::argv_type        argv_type;

      /// Default constructor, creates an uninitialised stream.
      basic_ipstream()
      : istream_type(NULL), pbase_type()
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling do_open() with the supplied
       * arguments.
       *
       * @param cmd   a string containing a shell command.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, pmode)
       */
      explicit
      basic_ipstream(const std::string& cmd, pmode mode = pstdout)
      : istream_type(NULL), pbase_type(cmd, readable(mode))
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling do_open() with the supplied
       * arguments.
       *
       * @param file  a string containing the pathname of a program to execute.
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      basic_ipstream( const std::string& file,
                      const argv_type& argv,
                      pmode mode = pstdout )
      : istream_type(NULL), pbase_type(file, argv, readable(mode))
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling
       * @c do_open(argv[0],argv,mode|pstdout)
       *
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      explicit
      basic_ipstream(const argv_type& argv, pmode mode = pstdout)
      : istream_type(NULL), pbase_type(argv.at(0), argv, readable(mode))
      { }

#if __cplusplus >= 201103L
      template<typename T>
        explicit
        basic_ipstream(std::initializer_list<T> args, pmode mode = pstdout)
        : basic_ipstream(argv_type(args.begin(), args.end()), mode)
        { }
#endif

      /**
       * @brief Destructor.
       *
       * Closes the stream and waits for the child to exit.
       */
      ~basic_ipstream()
      { }

      /**
       * @brief Start a process.
       *
       * Calls do_open( @a cmd , @a mode|pstdout ).
       *
       * @param cmd   a string containing a shell command.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, pmode)
       */
      void
      open(const std::string& cmd, pmode mode = pstdout)
      {
        this->do_open(cmd, readable(mode));
      }

      /**
       * @brief Start a process.
       *
       * Calls do_open( @a file , @a argv , @a mode|pstdout ).
       *
       * @param file  a string containing the pathname of a program to execute.
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      void
      open( const std::string& file,
            const argv_type& argv,
            pmode mode = pstdout )
      {
        this->do_open(file, argv, readable(mode));
      }

      /**
       * @brief Set streambuf to read from process' @c stdout.
       * @return  @c *this
       */
      basic_ipstream&
      out()
      {
        this->buf_.read_err(false);
        return *this;
      }

      /**
       * @brief Set streambuf to read from process' @c stderr.
       * @return  @c *this
       */
      basic_ipstream&
      err()
      {
        this->buf_.read_err(true);
        return *this;
      }
    };


  /**
   * @class basic_opstream
   * @brief Class template for Output PStreams.
   *
   * Writing to an open opstream writes to the standard input of the command;
   * the command's standard output is the same as that of the process that
   * created the pstream object, unless altered by the command itself.
   */

  template <typename CharT, typename Traits = std::char_traits<CharT> >
    class basic_opstream
    : public std::basic_ostream<CharT, Traits>
    , public pstream_common<CharT, Traits>
    , virtual public pstreams
    {
      typedef std::basic_ostream<CharT, Traits>     ostream_type;
      typedef pstream_common<CharT, Traits>         pbase_type;

      using pbase_type::buf_;  // declare name in this scope

    public:
      /// Type used to specify how to connect to the process.
      typedef typename pbase_type::pmode            pmode;

      /// Type used to hold the arguments for a command.
      typedef typename pbase_type::argv_type        argv_type;

      /// Default constructor, creates an uninitialised stream.
      basic_opstream()
      : ostream_type(NULL), pbase_type()
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling do_open() with the supplied
       * arguments.
       *
       * @param cmd   a string containing a shell command.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, pmode)
       */
      explicit
      basic_opstream(const std::string& cmd, pmode mode = pstdin)
      : ostream_type(NULL), pbase_type(cmd, mode|pstdin)
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling do_open() with the supplied
       * arguments.
       *
       * @param file  a string containing the pathname of a program to execute.
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      basic_opstream( const std::string& file,
                      const argv_type& argv,
                      pmode mode = pstdin )
      : ostream_type(NULL), pbase_type(file, argv, mode|pstdin)
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling
       * @c do_open(argv[0],argv,mode|pstdin)
       *
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      explicit
      basic_opstream(const argv_type& argv, pmode mode = pstdin)
      : ostream_type(NULL), pbase_type(argv.at(0), argv, mode|pstdin)
      { }

#if __cplusplus >= 201103L
      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * @param args  a list of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      template<typename T>
        explicit
        basic_opstream(std::initializer_list<T> args, pmode mode = pstdin)
        : basic_opstream(argv_type(args.begin(), args.end()), mode)
        { }
#endif

      /**
       * @brief Destructor
       *
       * Closes the stream and waits for the child to exit.
       */
      ~basic_opstream() { }

      /**
       * @brief Start a process.
       *
       * Calls do_open( @a cmd , @a mode|pstdin ).
       *
       * @param cmd   a string containing a shell command.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, pmode)
       */
      void
      open(const std::string& cmd, pmode mode = pstdin)
      {
        this->do_open(cmd, mode|pstdin);
      }

      /**
       * @brief Start a process.
       *
       * Calls do_open( @a file , @a argv , @a mode|pstdin ).
       *
       * @param file  a string containing the pathname of a program to execute.
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      void
      open( const std::string& file,
            const argv_type& argv,
            pmode mode = pstdin)
      {
        this->do_open(file, argv, mode|pstdin);
      }
    };


  /**
   * @class basic_pstream
   * @brief Class template for Bidirectional PStreams.
   *
   * Writing to a pstream opened with @c pmode @c pstdin writes to the
   * standard input of the command.
   * Reading from a pstream opened with @c pmode @c pstdout and/or @c pstderr
   * reads the command's standard output and/or standard error.
   * Any of the process' @c stdin, @c stdout or @c stderr that is not
   * connected to the pstream (as specified by the @c pmode)
   * will be the same as the process that created the pstream object,
   * unless altered by the command itself.
   */
  template <typename CharT, typename Traits = std::char_traits<CharT> >
    class basic_pstream
    : public std::basic_iostream<CharT, Traits>
    , public pstream_common<CharT, Traits>
    , virtual public pstreams
    {
      typedef std::basic_iostream<CharT, Traits>    iostream_type;
      typedef pstream_common<CharT, Traits>         pbase_type;

      using pbase_type::buf_;  // declare name in this scope

    public:
      /// Type used to specify how to connect to the process.
      typedef typename pbase_type::pmode            pmode;

      /// Type used to hold the arguments for a command.
      typedef typename pbase_type::argv_type        argv_type;

      /// Default constructor, creates an uninitialised stream.
      basic_pstream()
      : iostream_type(NULL), pbase_type()
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling do_open() with the supplied
       * arguments.
       *
       * @param cmd   a string containing a shell command.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, pmode)
       */
      explicit
      basic_pstream(const std::string& cmd, pmode mode = pstdout|pstdin)
      : iostream_type(NULL), pbase_type(cmd, mode)
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling do_open() with the supplied
       * arguments.
       *
       * @param file  a string containing the pathname of a program to execute.
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      basic_pstream( const std::string& file,
                     const argv_type& argv,
                     pmode mode = pstdout|pstdin )
      : iostream_type(NULL), pbase_type(file, argv, mode)
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling
       * @c do_open(argv[0],argv,mode)
       *
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      explicit
      basic_pstream(const argv_type& argv, pmode mode = pstdout|pstdin)
      : iostream_type(NULL), pbase_type(argv.at(0), argv, mode)
      { }

#if __cplusplus >= 201103L
      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * @param l     a list of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      template<typename T>
        explicit
        basic_pstream(std::initializer_list<T> l, pmode mode = pstdout|pstdin)
        : basic_pstream(argv_type(l.begin(), l.end()), mode)
        { }
#endif

      /**
       * @brief Destructor
       *
       * Closes the stream and waits for the child to exit.
       */
      ~basic_pstream() { }

      /**
       * @brief Start a process.
       *
       * Calls do_open( @a cnd , @a mode ).
       *
       * @param cmd   a string containing a shell command.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, pmode)
       */
      void
      open(const std::string& cmd, pmode mode = pstdout|pstdin)
      {
        this->do_open(cmd, mode);
      }

      /**
       * @brief Start a process.
       *
       * Calls do_open( @a file , @a argv , @a mode ).
       *
       * @param file  a string containing the pathname of a program to execute.
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      void
      open( const std::string& file,
            const argv_type& argv,
            pmode mode = pstdout|pstdin )
      {
        this->do_open(file, argv, mode);
      }

      /**
       * @brief Set streambuf to read from process' @c stdout.
       * @return  @c *this
       */
      basic_pstream&
      out()
      {
        this->buf_.read_err(false);
        return *this;
      }

      /**
       * @brief Set streambuf to read from process' @c stderr.
       * @return  @c *this
       */
      basic_pstream&
      err()
      {
        this->buf_.read_err(true);
        return *this;
      }
    };


  /**
   * @class basic_rpstream
   * @brief Class template for Restricted PStreams.
   *
   * Writing to an rpstream opened with @c pmode @c pstdin writes to the
   * standard input of the command.
   * It is not possible to read directly from an rpstream object, to use
   * an rpstream as in istream you must call either basic_rpstream::out()
   * or basic_rpstream::err(). This is to prevent accidental reads from
   * the wrong input source. If the rpstream was not opened with @c pmode
   * @c pstderr then the class cannot read the process' @c stderr, and
   * basic_rpstream::err() will return an istream that reads from the
   * process' @c stdout, and vice versa.
   * Reading from an rpstream opened with @c pmode @c pstdout and/or
   * @c pstderr reads the command's standard output and/or standard error.
   * Any of the process' @c stdin, @c stdout or @c stderr that is not
   * connected to the pstream (as specified by the @c pmode)
   * will be the same as the process that created the pstream object,
   * unless altered by the command itself.
   */

  template <typename CharT, typename Traits = std::char_traits<CharT> >
    class basic_rpstream
    : public std::basic_ostream<CharT, Traits>
    , private std::basic_istream<CharT, Traits>
    , private pstream_common<CharT, Traits>
    , virtual public pstreams
    {
      typedef std::basic_ostream<CharT, Traits>     ostream_type;
      typedef std::basic_istream<CharT, Traits>     istream_type;
      typedef pstream_common<CharT, Traits>         pbase_type;

      using pbase_type::buf_;  // declare name in this scope

    public:
      /// Type used to specify how to connect to the process.
      typedef typename pbase_type::pmode            pmode;

      /// Type used to hold the arguments for a command.
      typedef typename pbase_type::argv_type        argv_type;

      /// Default constructor, creates an uninitialised stream.
      basic_rpstream()
      : ostream_type(NULL), istream_type(NULL), pbase_type()
      { }

      /**
       * @brief  Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling do_open() with the supplied
       * arguments.
       *
       * @param cmd   a string containing a shell command.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, pmode)
       */
      explicit
      basic_rpstream(const std::string& cmd, pmode mode = pstdout|pstdin)
      : ostream_type(NULL) , istream_type(NULL) , pbase_type(cmd, mode)
      { }

      /**
       * @brief  Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling do_open() with the supplied
       * arguments.
       *
       * @param file a string containing the pathname of a program to execute.
       * @param argv a vector of argument strings passed to the new program.
       * @param mode the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      basic_rpstream( const std::string& file,
                      const argv_type& argv,
                      pmode mode = pstdout|pstdin )
      : ostream_type(NULL), istream_type(NULL), pbase_type(file, argv, mode)
      { }

      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * Initialises the stream buffer by calling
       * @c do_open(argv[0],argv,mode)
       *
       * @param argv  a vector of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      explicit
      basic_rpstream(const argv_type& argv, pmode mode = pstdout|pstdin)
      : ostream_type(NULL), istream_type(NULL),
        pbase_type(argv.at(0), argv, mode)
      { }

#if __cplusplus >= 201103L
      /**
       * @brief Constructor that initialises the stream by starting a process.
       *
       * @param l     a list of argument strings passed to the new program.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      template<typename T>
        explicit
        basic_rpstream(std::initializer_list<T> l, pmode mode = pstdout|pstdin)
        : basic_rpstream(argv_type(l.begin(), l.end()), mode)
        { }
#endif

      /// Destructor
      ~basic_rpstream() { }

      /**
       * @brief  Start a process.
       *
       * Calls do_open( @a cmd , @a mode ).
       *
       * @param cmd   a string containing a shell command.
       * @param mode  the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, pmode)
       */
      void
      open(const std::string& cmd, pmode mode = pstdout|pstdin)
      {
        this->do_open(cmd, mode);
      }

      /**
       * @brief  Start a process.
       *
       * Calls do_open( @a file , @a argv , @a mode ).
       *
       * @param file a string containing the pathname of a program to execute.
       * @param argv a vector of argument strings passed to the new program.
       * @param mode the I/O mode to use when opening the pipe.
       * @see   do_open(const std::string&, const argv_type&, pmode)
       */
      void
      open( const std::string& file,
            const argv_type& argv,
            pmode mode = pstdout|pstdin )
      {
        this->do_open(file, argv, mode);
      }

      /**
       * @brief  Obtain a reference to the istream that reads
       *         the process' @c stdout.
       * @return @c *this
       */
      istream_type&
      out()
      {
        this->buf_.read_err(false);
        return *this;
      }

      /**
       * @brief  Obtain a reference to the istream that reads
       *         the process' @c stderr.
       * @return @c *this
       */
      istream_type&
      err()
      {
        this->buf_.read_err(true);
        return *this;
      }
    };


  /// Type definition for common template specialisation.
  typedef basic_pstreambuf<char> pstreambuf;
  /// Type definition for common template specialisation.
  typedef basic_ipstream<char> ipstream;
  /// Type definition for common template specialisation.
  typedef basic_opstream<char> opstream;
  /// Type definition for common template specialisation.
  typedef basic_pstream<char> pstream;
  /// Type definition for common template specialisation.
  typedef basic_rpstream<char> rpstream;


  /**
   * When inserted into an output pstream the manipulator calls
   * basic_pstreambuf<C,T>::peof() to close the output pipe,
   * causing the child process to receive the end-of-file indicator
   * on subsequent reads from its @c stdin stream.
   *
   * @brief   Manipulator to close the pipe connected to the process' stdin.
   * @param   s  An output PStream class.
   * @return  The stream object the manipulator was invoked on.
   * @see     basic_pstreambuf<C,T>::peof()
   * @relates basic_opstream basic_pstream basic_rpstream
   */
  template <typename C, typename T>
    inline std::basic_ostream<C,T>&
    peof(std::basic_ostream<C,T>& s)
    {
      typedef basic_pstreambuf<C,T> pstreambuf_type;
      if (pstreambuf_type* p = dynamic_cast<pstreambuf_type*>(s.rdbuf()))
        p->peof();
      return s;
    }


  /*
   * member definitions for pstreambuf
   */


  /**
   * @class basic_pstreambuf
   * Provides underlying streambuf functionality for the PStreams classes.
   */

  /** Creates an uninitialised stream buffer. */
  template <typename C, typename T>
    inline
    basic_pstreambuf<C,T>::basic_pstreambuf()
    : ppid_(-1)   // initialise to -1 to indicate no process run yet.
    , wpipe_(-1)
    , wbuffer_(NULL)
    , rsrc_(rsrc_out)
    , status_(-1)
    , error_(0)
    {
      init_rbuffers();
    }

  /**
   * Initialises the stream buffer by calling open() with the supplied
   * arguments.
   *
   * @param cmd   a string containing a shell command.
   * @param mode  the I/O mode to use when opening the pipe.
   * @see   open()
   */
  template <typename C, typename T>
    inline
    basic_pstreambuf<C,T>::basic_pstreambuf(const std::string& cmd, pmode mode)
    : ppid_(-1)   // initialise to -1 to indicate no process run yet.
    , wpipe_(-1)
    , wbuffer_(NULL)
    , rsrc_(rsrc_out)
    , status_(-1)
    , error_(0)
    {
      init_rbuffers();
      open(cmd, mode);
    }

  /**
   * Initialises the stream buffer by calling open() with the supplied
   * arguments.
   *
   * @param file  a string containing the name of a program to execute.
   * @param argv  a vector of argument strings passsed to the new program.
   * @param mode  the I/O mode to use when opening the pipe.
   * @see   open()
   */
  template <typename C, typename T>
    inline
    basic_pstreambuf<C,T>::basic_pstreambuf( const std::string& file,
                                             const argv_type& argv,
                                             pmode mode )
    : ppid_(-1)   // initialise to -1 to indicate no process run yet.
    , wpipe_(-1)
    , wbuffer_(NULL)
    , rsrc_(rsrc_out)
    , status_(-1)
    , error_(0)
    {
      init_rbuffers();
      open(file, argv, mode);
    }

  /**
   * Closes the stream by calling close().
   * @see close()
   */
  template <typename C, typename T>
    inline
    basic_pstreambuf<C,T>::~basic_pstreambuf()
    {
      close();
    }

  /**
   * Starts a new process by passing @a command to the shell (/bin/sh)
   * and opens pipes to the process with the specified @a mode.
   *
   * If @a mode contains @c pstdout the initial read source will be
   * the child process' stdout, otherwise if @a mode  contains @c pstderr
   * the initial read source will be the child's stderr.
   *
   * Will duplicate the actions of  the  shell  in searching for an
   * executable file if the specified file name does not contain a slash (/)
   * character.
   *
   * @warning
   * There is no way to tell whether the shell command succeeded, this
   * function will always succeed unless resource limits (such as
   * memory usage, or number of processes or open files) are exceeded.
   * This means is_open() will return true even if @a command cannot
   * be executed.
   * Use pstreambuf::open(const std::string&, const argv_type&, pmode)
   * if you need to know whether the command failed to execute.
   *
   * @param   command  a string containing a shell command.
   * @param   mode     a bitwise OR of one or more of @c out, @c in, @c err.
   * @return  NULL if the shell could not be started or the
   *          pipes could not be opened, @c this otherwise.
   * @see     <b>execl</b>(3)
   */
  template <typename C, typename T>
    basic_pstreambuf<C,T>*
    basic_pstreambuf<C,T>::open(const std::string& command, pmode mode)
    {
      const char * shell_path = "/bin/sh";
#if 0
      const std::string argv[] = { "sh", "-c", command };
      return this->open(shell_path, argv_type(argv, argv+3), mode);
#else
      basic_pstreambuf<C,T>* ret = NULL;

      if (!is_open())
      {
        switch(fork(mode))
        {
        case 0 :
          // this is the new process, exec command
          ::execl(shell_path, "sh", "-c", command.c_str(), (char*)NULL);

          // can only reach this point if exec() failed

          // parent can get exit code from waitpid()
          ::_exit(errno);
          // using std::exit() would make static dtors run twice

        case -1 :
          // couldn't fork, error already handled in pstreambuf::fork()
          break;

        default :
          // this is the parent process
          // activate buffers
          create_buffers(mode);
          ret = this;
        }
      }
      return ret;
#endif
    }

  /**
   * @brief  Helper function to close a file descriptor.
   *
   * Inspects @a fd and calls <b>close</b>(3) if it has a non-negative value.
   *
   * @param   fd  a file descriptor.
   * @relates basic_pstreambuf
   */
  inline void
  close_fd(pstreams::fd_type& fd)
  {
    if (fd >= 0 && ::close(fd) == 0)
      fd = -1;
  }

  /**
   * @brief  Helper function to close an array of file descriptors.
   *
   * Calls @c close_fd() on each member of the array.
   * The length of the array is determined automatically by
   * template argument deduction to avoid errors.
   *
   * @param   fds  an array of file descriptors.
   * @relates basic_pstreambuf
   */
  template <int N>
    inline void
    close_fd_array(pstreams::fd_type (&fds)[N])
    {
      for (std::size_t i = 0; i < N; ++i)
        close_fd(fds[i]);
    }

  /**
   * Starts a new process by executing @a file with the arguments in
   * @a argv and opens pipes to the process with the specified @a mode.
   *
   * By convention @c argv[0] should be the file name of the file being
   * executed.
   *
   * If @a mode contains @c pstdout the initial read source will be
   * the child process' stdout, otherwise if @a mode  contains @c pstderr
   * the initial read source will be the child's stderr.
   *
   * Will duplicate the actions of  the  shell  in searching for an
   * executable file if the specified file name does not contain a slash (/)
   * character.
   *
   * Iff @a file is successfully executed then is_open() will return true.
   * Otherwise, pstreambuf::error() can be used to obtain the value of
   * @c errno that was set by <b>execvp</b>(3) in the child process.
   *
   * The exit status of the new process will be returned by
   * pstreambuf::status() after pstreambuf::exited() returns true.
   *
   * @param   file  a string containing the pathname of a program to execute.
   * @param   argv  a vector of argument strings passed to the new program.
   * @param   mode  a bitwise OR of one or more of @c out, @c in and @c err.
   * @return  NULL if a pipe could not be opened or if the program could
   *          not be executed, @c this otherwise.
   * @see     <b>execvp</b>(3)
   */
  template <typename C, typename T>
    basic_pstreambuf<C,T>*
    basic_pstreambuf<C,T>::open( const std::string& file,
                                 const argv_type& argv,
                                 pmode mode )
    {
      basic_pstreambuf<C,T>* ret = NULL;

      if (!is_open())
      {
        // constants for read/write ends of pipe
        enum { RD, WR };

        // open another pipe and set close-on-exec
        fd_type ck_exec[] = { -1, -1 };
        if (-1 == ::pipe(ck_exec)
            || -1 == ::fcntl(ck_exec[RD], F_SETFD, FD_CLOEXEC)
            || -1 == ::fcntl(ck_exec[WR], F_SETFD, FD_CLOEXEC))
        {
          error_ = errno;
          close_fd_array(ck_exec);
        }
        else
        {
          switch(fork(mode))
          {
          case 0 :
            // this is the new process, exec command
            {
              char** arg_v = new char*[argv.size()+1];
              for (std::size_t i = 0; i < argv.size(); ++i)
              {
                const std::string& src = argv[i];
                char*& dest = arg_v[i];
                dest = new char[src.size()+1];
                dest[ src.copy(dest, src.size()) ] = '\0';
              }
              arg_v[argv.size()] = NULL;

              ::execvp(file.c_str(), arg_v);

              // can only reach this point if exec() failed

              // parent can get error code from ck_exec pipe
              error_ = errno;

              while (::write(ck_exec[WR], &error_, sizeof(error_)) == -1
                  && errno == EINTR)
              { }

              ::close(ck_exec[WR]);
              ::close(ck_exec[RD]);

              ::_exit(error_);
              // using std::exit() would make static dtors run twice
            }

          case -1 :
            // couldn't fork, error already handled in pstreambuf::fork()
            close_fd_array(ck_exec);
            break;

          default :
            // this is the parent process

            // check child called exec() successfully
            ::close(ck_exec[WR]);
            switch (::read(ck_exec[RD], &error_, sizeof(error_)))
            {
            case 0:
              // activate buffers
              create_buffers(mode);
              ret = this;
              break;
            case -1:
              error_ = errno;
              break;
            default:
              // error_ contains error code from child
              // call wait() to clean up and set ppid_ to 0
              this->wait();
              break;
            }
            ::close(ck_exec[RD]);
          }
        }
      }
      return ret;
    }

  /**
   * Creates pipes as specified by @a mode and calls @c fork() to create
   * a new process. If the fork is successful the parent process stores
   * the child's PID and the opened pipes and the child process replaces
   * its standard streams with the opened pipes.
   *
   * If an error occurs the error code will be set to one of the possible
   * errors for @c pipe() or @c fork().
   * See your system's documentation for these error codes.
   *
   * @param   mode  an OR of pmodes specifying which of the child's
   *                standard streams to connect to.
   * @return  On success the PID of the child is returned in the parent's
   *          context and zero is returned in the child's context.
   *          On error -1 is returned and the error code is set appropriately.
   */
  template <typename C, typename T>
    pid_t
    basic_pstreambuf<C,T>::fork(pmode mode)
    {
      pid_t pid = -1;

      // Three pairs of file descriptors, for pipes connected to the
      // process' stdin, stdout and stderr
      // (stored in a single array so close_fd_array() can close all at once)
      fd_type fd[] = { -1, -1, -1, -1, -1, -1 };
      fd_type* const pin = fd;
      fd_type* const pout = fd+2;
      fd_type* const perr = fd+4;

      // constants for read/write ends of pipe
      enum { RD, WR };

      // N.B.
      // For the pstreambuf pin is an output stream and
      // pout and perr are input streams.

      if (!error_ && mode&pstdin && ::pipe(pin))
        error_ = errno;

      if (!error_ && mode&pstdout && ::pipe(pout))
        error_ = errno;

      if (!error_ && mode&pstderr && ::pipe(perr))
        error_ = errno;

      if (!error_)
      {
        pid = ::fork();
        switch (pid)
        {
          case 0 :
          {
            // this is the new process

            // for each open pipe close one end and redirect the
            // respective standard stream to the other end

            if (*pin >= 0)
            {
              ::close(pin[WR]);
              ::dup2(pin[RD], STDIN_FILENO);
              ::close(pin[RD]);
            }
            if (*pout >= 0)
            {
              ::close(pout[RD]);
              ::dup2(pout[WR], STDOUT_FILENO);
              ::close(pout[WR]);
            }
            if (*perr >= 0)
            {
              ::close(perr[RD]);
              ::dup2(perr[WR], STDERR_FILENO);
              ::close(perr[WR]);
            }

#ifdef _POSIX_JOB_CONTROL
            if (mode&newpg)
              ::setpgid(0, 0); // Change to a new process group
#endif

            break;
          }
          case -1 :
          {
            // couldn't fork for some reason
            error_ = errno;
            // close any open pipes
            close_fd_array(fd);
            break;
          }
          default :
          {
            // this is the parent process, store process' pid
            ppid_ = pid;

            // store one end of open pipes and close other end
            if (*pin >= 0)
            {
              wpipe_ = pin[WR];
              ::close(pin[RD]);
            }
            if (*pout >= 0)
            {
              rpipe_[rsrc_out] = pout[RD];
              ::close(pout[WR]);
            }
            if (*perr >= 0)
            {
              rpipe_[rsrc_err] = perr[RD];
              ::close(perr[WR]);
            }
          }
        }
      }
      else
      {
        // close any pipes we opened before failure
        close_fd_array(fd);
      }
      return pid;
    }

  /**
   * Closes all pipes and calls wait() to wait for the process to finish.
   * If an error occurs the error code will be set to one of the possible
   * errors for @c waitpid().
   * See your system's documentation for these errors.
   *
   * @return  @c this on successful close or @c NULL if there is no
   *          process to close or if an error occurs.
   */
  template <typename C, typename T>
    basic_pstreambuf<C,T>*
    basic_pstreambuf<C,T>::close()
    {
      const bool running = is_open();

      sync(); // this might call wait() and reap the child process

      // rather than trying to work out whether or not we need to clean up
      // just do it anyway, all cleanup functions are safe to call twice.

      destroy_buffers(pstdin|pstdout|pstderr);

      // close pipes before wait() so child gets EOF/SIGPIPE
      close_fd(wpipe_);
      close_fd_array(rpipe_);

      do
      {
        error_ = 0;
      } while (wait() == -1 && error() == EINTR);

      return running ? this : NULL;
    }

  /**
   *  Called on construction to initialise the arrays used for reading.
   */
  template <typename C, typename T>
    inline void
    basic_pstreambuf<C,T>::init_rbuffers()
    {
      rpipe_[rsrc_out] = rpipe_[rsrc_err] = -1;
      rbuffer_[rsrc_out] = rbuffer_[rsrc_err] = NULL;
      rbufstate_[0] = rbufstate_[1] = rbufstate_[2] = NULL;
    }

  template <typename C, typename T>
    void
    basic_pstreambuf<C,T>::create_buffers(pmode mode)
    {
      if (mode & pstdin)
      {
        delete[] wbuffer_;
        wbuffer_ = new char_type[bufsz];
        this->setp(wbuffer_, wbuffer_ + bufsz);
      }
      if (mode & pstdout)
      {
        delete[] rbuffer_[rsrc_out];
        rbuffer_[rsrc_out] = new char_type[bufsz];
        rsrc_ = rsrc_out;
        this->setg(rbuffer_[rsrc_out] + pbsz, rbuffer_[rsrc_out] + pbsz,
            rbuffer_[rsrc_out] + pbsz);
      }
      if (mode & pstderr)
      {
        delete[] rbuffer_[rsrc_err];
        rbuffer_[rsrc_err] = new char_type[bufsz];
        if (!(mode & pstdout))
        {
          rsrc_ = rsrc_err;
          this->setg(rbuffer_[rsrc_err] + pbsz, rbuffer_[rsrc_err] + pbsz,
              rbuffer_[rsrc_err] + pbsz);
        }
      }
    }

  template <typename C, typename T>
    void
    basic_pstreambuf<C,T>::destroy_buffers(pmode mode)
    {
      if (mode & pstdin)
      {
        this->setp(NULL, NULL);
        delete[] wbuffer_;
        wbuffer_ = NULL;
      }
      if (mode & pstdout)
      {
        if (rsrc_ == rsrc_out)
          this->setg(NULL, NULL, NULL);
        delete[] rbuffer_[rsrc_out];
        rbuffer_[rsrc_out] = NULL;
      }
      if (mode & pstderr)
      {
        if (rsrc_ == rsrc_err)
          this->setg(NULL, NULL, NULL);
        delete[] rbuffer_[rsrc_err];
        rbuffer_[rsrc_err] = NULL;
      }
    }

  template <typename C, typename T>
    typename basic_pstreambuf<C,T>::buf_read_src
    basic_pstreambuf<C,T>::switch_read_buffer(buf_read_src src)
    {
      if (rsrc_ != src)
      {
        char_type* tmpbufstate[] = {this->eback(), this->gptr(), this->egptr()};
        this->setg(rbufstate_[0], rbufstate_[1], rbufstate_[2]);
        for (std::size_t i = 0; i < 3; ++i)
          rbufstate_[i] = tmpbufstate[i];
        rsrc_ = src;
      }
      return rsrc_;
    }

  /**
   * Suspends execution and waits for the associated process to exit, or
   * until a signal is delivered whose action is to terminate the current
   * process or to call a signal handling function. If the process has
   * already exited (i.e. it is a "zombie" process) then wait() returns
   * immediately.  Waiting for the child process causes all its system
   * resources to be freed.
   *
   * error() will return EINTR if wait() is interrupted by a signal.
   *
   * @param   nohang  true to return immediately if the process has not exited.
   * @return  1 if the process has exited and wait() has not yet been called.
   *          0 if @a nohang is true and the process has not exited yet.
   *          -1 if no process has been started or if an error occurs,
   *          in which case the error can be found using error().
   */
  template <typename C, typename T>
    int
    basic_pstreambuf<C,T>::wait(bool nohang)
    {
      int child_exited = -1;
      if (is_open())
      {
        int exit_status;
        switch(::waitpid(ppid_, &exit_status, nohang ? WNOHANG : 0))
        {
          case 0 :
            // nohang was true and process has not exited
            child_exited = 0;
            break;
          case -1 :
            error_ = errno;
            break;
          default :
            // process has exited
            ppid_ = 0;
            status_ = exit_status;
            child_exited = 1;
            // Close wpipe, would get SIGPIPE if we used it.
            destroy_buffers(pstdin);
            close_fd(wpipe_);
            // Must free read buffers and pipes on destruction
            // or next call to open()/close()
            break;
        }
      }
      return child_exited;
    }

  /**
   * Sends the specified signal to the process.  A signal can be used to
   * terminate a child process that would not exit otherwise.
   *
   * If an error occurs the error code will be set to one of the possible
   * errors for @c kill().  See your system's documentation for these errors.
   *
   * @param   signal  A signal to send to the child process.
   * @return  @c this or @c NULL if @c kill() fails.
   */
  template <typename C, typename T>
    inline basic_pstreambuf<C,T>*
    basic_pstreambuf<C,T>::kill(int signal)
    {
      basic_pstreambuf<C,T>* ret = NULL;
      if (is_open())
      {
        if (::kill(ppid_, signal))
          error_ = errno;
        else
        {
#if 0
          // TODO call exited() to check for exit and clean up? leave to user?
          if (signal==SIGTERM || signal==SIGKILL)
            this->exited();
#endif
          ret = this;
        }
      }
      return ret;
    }

  /**
   * Sends the specified signal to the process group of the child process.
   * A signal can be used to terminate a child process that would not exit
   * otherwise, or to kill the process and its own children.
   *
   * If an error occurs the error code will be set to one of the possible
   * errors for @c getpgid() or @c kill().  See your system's documentation
   * for these errors. If the child is in the current process group then
   * NULL will be returned and the error code set to EPERM.
   *
   * @param   signal  A signal to send to the child process.
   * @return  @c this on success or @c NULL on failure.
   */
  template <typename C, typename T>
    inline basic_pstreambuf<C,T>*
    basic_pstreambuf<C,T>::killpg(int signal)
    {
      basic_pstreambuf<C,T>* ret = NULL;
#ifdef _POSIX_JOB_CONTROL
      if (is_open())
      {
        pid_t pgid = ::getpgid(ppid_);
        if (pgid == -1)
          error_ = errno;
        else if (pgid == ::getpgrp())
          error_ = EPERM;  // Don't commit suicide
        else if (::killpg(pgid, signal))
          error_ = errno;
        else
          ret = this;
      }
#else
      error_ = ENOTSUP;
#endif
      return ret;
    }

  /**
   *  This function can call pstreambuf::wait() and so may change the
   *  object's state if the child process has already exited.
   *
   *  @return  True if the associated process has exited, false otherwise.
   *  @see     basic_pstreambuf<C,T>::wait()
   */
  template <typename C, typename T>
    inline bool
    basic_pstreambuf<C,T>::exited()
    {
      return ppid_ == 0 || wait(true)==1;
    }


  /**
   *  @return  The exit status of the child process, or -1 if wait()
   *           has not yet been called to wait for the child to exit.
   *  @see     basic_pstreambuf<C,T>::wait()
   */
  template <typename C, typename T>
    inline int
    basic_pstreambuf<C,T>::status() const
    {
      return status_;
    }

  /**
   *  @return  The error code of the most recently failed operation, or zero.
   */
  template <typename C, typename T>
    inline int
    basic_pstreambuf<C,T>::error() const
    {
      return error_;
    }

  /**
   *  Closes the output pipe, causing the child process to receive the
   *  end-of-file indicator on subsequent reads from its @c stdin stream.
   */
  template <typename C, typename T>
    inline void
    basic_pstreambuf<C,T>::peof()
    {
      sync();
      destroy_buffers(pstdin);
      close_fd(wpipe_);
    }

  /**
   * Unlike pstreambuf::exited(), this function will not call wait() and
   * so will not change the object's state.  This means that once a child
   * process is executed successfully this function will continue to
   * return true even after the process exits (until wait() is called.)
   *
   * @return  true if a previous call to open() succeeded and wait() has
   *          not been called and determined that the process has exited,
   *          false otherwise.
   */
  template <typename C, typename T>
    inline bool
    basic_pstreambuf<C,T>::is_open() const
    {
      return ppid_ > 0;
    }

  /**
   * Toggle the stream used for reading. If @a readerr is @c true then the
   * process' @c stderr output will be used for subsequent extractions, if
   * @a readerr is false the the process' stdout will be used.
   * @param   readerr  @c true to read @c stderr, @c false to read @c stdout.
   * @return  @c true if the requested stream is open and will be used for
   *          subsequent extractions, @c false otherwise.
   */
  template <typename C, typename T>
    inline bool
    basic_pstreambuf<C,T>::read_err(bool readerr)
    {
      buf_read_src src = readerr ? rsrc_err : rsrc_out;
      if (rpipe_[src]>=0)
      {
        switch_read_buffer(src);
        return true;
      }
      return false;
    }

  /**
   * Called when the internal character buffer is not present or is full,
   * to transfer the buffer contents to the pipe.
   *
   * @param   c  a character to be written to the pipe.
   * @return  @c traits_type::eof() if an error occurs, otherwise if @a c
   *          is not equal to @c traits_type::eof() it will be buffered and
   *          a value other than @c traits_type::eof() returned to indicate
   *          success.
   */
  template <typename C, typename T>
    typename basic_pstreambuf<C,T>::int_type
    basic_pstreambuf<C,T>::overflow(int_type c)
    {
      if (!empty_buffer())
        return traits_type::eof();
      else if (!traits_type::eq_int_type(c, traits_type::eof()))
        return this->sputc(c);
      else
        return traits_type::not_eof(c);
    }


  template <typename C, typename T>
    int
    basic_pstreambuf<C,T>::sync()
    {
      return !exited() && empty_buffer() ? 0 : -1;
    }

  /**
   * @param   s  character buffer.
   * @param   n  buffer length.
   * @return  the number of characters written.
   */
  template <typename C, typename T>
    std::streamsize
    basic_pstreambuf<C,T>::xsputn(const char_type* s, std::streamsize n)
    {
      std::streamsize done = 0;
      while (done < n)
      {
        if (std::streamsize nbuf = this->epptr() - this->pptr())
        {
          nbuf = std::min(nbuf, n - done);
          traits_type::copy(this->pptr(), s + done, nbuf);
          this->pbump(nbuf);
          done += nbuf;
        }
        else if (!empty_buffer())
          break;
      }
      return done;
    }

  /**
   * @return  true if the buffer was emptied, false otherwise.
   */
  template <typename C, typename T>
    bool
    basic_pstreambuf<C,T>::empty_buffer()
    {
      const std::streamsize count = this->pptr() - this->pbase();
      if (count > 0)
      {
        const std::streamsize written = this->write(this->wbuffer_, count);
        if (written > 0)
        {
          if (const std::streamsize unwritten = count - written)
            traits_type::move(this->pbase(), this->pbase()+written, unwritten);
          this->pbump(-written);
          return true;
        }
      }
      return false;
    }

  /**
   * Called when the internal character buffer is is empty, to re-fill it
   * from the pipe.
   *
   * @return The first available character in the buffer,
   * or @c traits_type::eof() in case of failure.
   */
  template <typename C, typename T>
    typename basic_pstreambuf<C,T>::int_type
    basic_pstreambuf<C,T>::underflow()
    {
      if (this->gptr() < this->egptr() || fill_buffer())
        return traits_type::to_int_type(*this->gptr());
      else
        return traits_type::eof();
    }

  /**
   * Attempts to make @a c available as the next character to be read by
   * @c sgetc().
   *
   * @param   c   a character to make available for extraction.
   * @return  @a c if the character can be made available,
   *          @c traits_type::eof() otherwise.
   */
  template <typename C, typename T>
    typename basic_pstreambuf<C,T>::int_type
    basic_pstreambuf<C,T>::pbackfail(int_type c)
    {
      if (this->gptr() != this->eback())
      {
        this->gbump(-1);
        if (!traits_type::eq_int_type(c, traits_type::eof()))
          *this->gptr() = traits_type::to_char_type(c);
        return traits_type::not_eof(c);
      }
      else
         return traits_type::eof();
    }

  template <typename C, typename T>
    std::streamsize
    basic_pstreambuf<C,T>::showmanyc()
    {
      int avail = 0;
      if (sizeof(char_type) == 1)
        avail = fill_buffer(true) ? this->egptr() - this->gptr() : -1;
#ifdef FIONREAD
      else
      {
        if (::ioctl(rpipe(), FIONREAD, &avail) == -1)
          avail = -1;
        else if (avail)
          avail /= sizeof(char_type);
      }
#endif
      return std::streamsize(avail);
    }

  /**
   * @return  true if the buffer was filled, false otherwise.
   */
  template <typename C, typename T>
    bool
    basic_pstreambuf<C,T>::fill_buffer(bool non_blocking)
    {
      const std::streamsize pb1 = this->gptr() - this->eback();
      const std::streamsize pb2 = pbsz;
      const std::streamsize npb = std::min(pb1, pb2);

      char_type* const rbuf = rbuffer();

      if (npb)
        traits_type::move(rbuf + pbsz - npb, this->gptr() - npb, npb);

      std::streamsize rc = -1;

      if (non_blocking)
      {
        const int flags = ::fcntl(rpipe(), F_GETFL);
        if (flags != -1)
        {
          const bool blocking = !(flags & O_NONBLOCK);
          if (blocking)
            ::fcntl(rpipe(), F_SETFL, flags | O_NONBLOCK);  // set non-blocking

          error_ = 0;
          rc = read(rbuf + pbsz, bufsz - pbsz);

          if (rc == -1 && error_ == EAGAIN)  // nothing available
            rc = 0;
          else if (rc == 0)  // EOF
            rc = -1;

          if (blocking)
            ::fcntl(rpipe(), F_SETFL, flags); // restore
        }
      }
      else
        rc = read(rbuf + pbsz, bufsz - pbsz);

      if (rc > 0 || (rc == 0 && non_blocking))
      {
        this->setg( rbuf + pbsz - npb,
                    rbuf + pbsz,
                    rbuf + pbsz + rc );
        return true;
      }
      else
      {
        this->setg(NULL, NULL, NULL);
        return false;
      }
    }

  /**
   * Writes up to @a n characters to the pipe from the buffer @a s.
   *
   * @param   s  character buffer.
   * @param   n  buffer length.
   * @return  the number of characters written.
   */
  template <typename C, typename T>
    inline std::streamsize
    basic_pstreambuf<C,T>::write(const char_type* s, std::streamsize n)
    {
      std::streamsize nwritten = 0;
      if (wpipe() >= 0)
      {
        nwritten = ::write(wpipe(), s, n * sizeof(char_type));
        if (nwritten == -1)
          error_ = errno;
        else
          nwritten /= sizeof(char_type);
      }
      return nwritten;
    }

  /**
   * Reads up to @a n characters from the pipe to the buffer @a s.
   *
   * @param   s  character buffer.
   * @param   n  buffer length.
   * @return  the number of characters read.
   */
  template <typename C, typename T>
    inline std::streamsize
    basic_pstreambuf<C,T>::read(char_type* s, std::streamsize n)
    {
      std::streamsize nread = 0;
      if (rpipe() >= 0)
      {
        nread = ::read(rpipe(), s, n * sizeof(char_type));
        if (nread == -1)
          error_ = errno;
        else
          nread /= sizeof(char_type);
      }
      return nread;
    }

  /** @return a reference to the output file descriptor */
  template <typename C, typename T>
    inline pstreams::fd_type&
    basic_pstreambuf<C,T>::wpipe()
    {
      return wpipe_;
    }

  /** @return a reference to the active input file descriptor */
  template <typename C, typename T>
    inline pstreams::fd_type&
    basic_pstreambuf<C,T>::rpipe()
    {
      return rpipe_[rsrc_];
    }

  /** @return a reference to the specified input file descriptor */
  template <typename C, typename T>
    inline pstreams::fd_type&
    basic_pstreambuf<C,T>::rpipe(buf_read_src which)
    {
      return rpipe_[which];
    }

  /** @return a pointer to the start of the active input buffer area. */
  template <typename C, typename T>
    inline typename basic_pstreambuf<C,T>::char_type*
    basic_pstreambuf<C,T>::rbuffer()
    {
      return rbuffer_[rsrc_];
    }


  /*
   * member definitions for pstream_common
   */

  /**
   * @class pstream_common
   * Abstract Base Class providing common functionality for basic_ipstream,
   * basic_opstream and basic_pstream.
   * pstream_common manages the basic_pstreambuf stream buffer that is used
   * by the derived classes to initialise an iostream class.
   */

  /** Creates an uninitialised stream. */
  template <typename C, typename T>
    inline
    pstream_common<C,T>::pstream_common()
    : std::basic_ios<C,T>(NULL)
    , command_()
    , buf_()
    {
      this->std::basic_ios<C,T>::rdbuf(&buf_);
    }

  /**
   * Initialises the stream buffer by calling
   * do_open( @a command , @a mode )
   *
   * @param cmd   a string containing a shell command.
   * @param mode  the I/O mode to use when opening the pipe.
   * @see   do_open(const std::string&, pmode)
   */
  template <typename C, typename T>
    inline
    pstream_common<C,T>::pstream_common(const std::string& cmd, pmode mode)
    : std::basic_ios<C,T>(NULL)
    , command_(cmd)
    , buf_()
    {
      this->std::basic_ios<C,T>::rdbuf(&buf_);
      do_open(cmd, mode);
    }

  /**
   * Initialises the stream buffer by calling
   * do_open( @a file , @a argv , @a mode )
   *
   * @param file  a string containing the pathname of a program to execute.
   * @param argv  a vector of argument strings passed to the new program.
   * @param mode  the I/O mode to use when opening the pipe.
   * @see do_open(const std::string&, const argv_type&, pmode)
   */
  template <typename C, typename T>
    inline
    pstream_common<C,T>::pstream_common( const std::string& file,
                                         const argv_type& argv,
                                         pmode mode )
    : std::basic_ios<C,T>(NULL)
    , command_(file)
    , buf_()
    {
      this->std::basic_ios<C,T>::rdbuf(&buf_);
      do_open(file, argv, mode);
    }

  /**
   * This is a pure virtual function to make @c pstream_common abstract.
   * Because it is the destructor it will be called by derived classes
   * and so must be defined.  It is also protected, to discourage use of
   * the PStreams classes through pointers or references to the base class.
   *
   * @sa If defining a pure virtual seems odd you should read
   * http://www.gotw.ca/gotw/031.htm (and the rest of the site as well!)
   */
  template <typename C, typename T>
    inline
    pstream_common<C,T>::~pstream_common()
    {
    }

  /**
   * Calls rdbuf()->open( @a command , @a mode )
   * and sets @c failbit on error.
   *
   * @param cmd   a string containing a shell command.
   * @param mode  the I/O mode to use when opening the pipe.
   * @see   basic_pstreambuf::open(const std::string&, pmode)
   */
  template <typename C, typename T>
    inline void
    pstream_common<C,T>::do_open(const std::string& cmd, pmode mode)
    {
      if (!buf_.open((command_=cmd), mode))
        this->setstate(std::ios_base::failbit);
    }

  /**
   * Calls rdbuf()->open( @a file, @a  argv, @a mode )
   * and sets @c failbit on error.
   *
   * @param file  a string containing the pathname of a program to execute.
   * @param argv  a vector of argument strings passed to the new program.
   * @param mode  the I/O mode to use when opening the pipe.
   * @see   basic_pstreambuf::open(const std::string&, const argv_type&, pmode)
   */
  template <typename C, typename T>
    inline void
    pstream_common<C,T>::do_open( const std::string& file,
                                  const argv_type& argv,
                                  pmode mode )
    {
      if (!buf_.open((command_=file), argv, mode))
        this->setstate(std::ios_base::failbit);
    }

  /** Calls rdbuf->close() and sets @c failbit on error. */
  template <typename C, typename T>
    inline void
    pstream_common<C,T>::close()
    {
      if (!buf_.close())
        this->setstate(std::ios_base::failbit);
    }

  /**
   * @return  rdbuf()->is_open().
   * @see     basic_pstreambuf::is_open()
   */
  template <typename C, typename T>
    inline bool
    pstream_common<C,T>::is_open() const
    {
      return buf_.is_open();
    }

  /** @return a string containing the command used to initialise the stream. */
  template <typename C, typename T>
    inline const std::string&
    pstream_common<C,T>::command() const
    {
      return command_;
    }

  /** @return a pointer to the private stream buffer member. */
  // TODO  document behaviour if buffer replaced.
  template <typename C, typename T>
    inline typename pstream_common<C,T>::streambuf_type*
    pstream_common<C,T>::rdbuf() const
    {
      return const_cast<streambuf_type*>(&buf_);
    }


#if REDI_EVISCERATE_PSTREAMS
  /**
   * @def REDI_EVISCERATE_PSTREAMS
   * If this macro has a non-zero value then certain internals of the
   * @c basic_pstreambuf template class are exposed. In general this is
   * a Bad Thing, as the internal implementation is largely undocumented
   * and may be subject to change at any time, so this feature is only
   * provided because it might make PStreams useful in situations where
   * it is necessary to do Bad Things.
   */

  /**
   * @warning  This function exposes the internals of the stream buffer and
   *           should be used with caution. It is the caller's responsibility
   *           to flush streams etc. in order to clear any buffered data.
   *           The POSIX.1 function <b>fdopen</b>(3) is used to obtain the
   *           @c FILE pointers from the streambuf's private file descriptor
   *           members so consult your system's documentation for
   *           <b>fdopen</b>(3).
   *
   * @param   in    A FILE* that will refer to the process' stdin.
   * @param   out   A FILE* that will refer to the process' stdout.
   * @param   err   A FILE* that will refer to the process' stderr.
   * @return  An OR of zero or more of @c pstdin, @c pstdout, @c pstderr.
   *
   * For each open stream shared with the child process a @c FILE* is
   * obtained and assigned to the corresponding parameter. For closed
   * streams @c NULL is assigned to the parameter.
   * The return value can be tested to see which parameters should be
   * @c !NULL by masking with the corresponding @c pmode value.
   *
   * @see <b>fdopen</b>(3)
   */
  template <typename C, typename T>
    std::size_t
    basic_pstreambuf<C,T>::fopen(FILE*& in, FILE*& out, FILE*& err)
    {
      in = out = err = NULL;
      std::size_t open_files = 0;
      if (wpipe() > -1)
      {
        if ((in = ::fdopen(wpipe(), "w")))
        {
            open_files |= pstdin;
        }
      }
      if (rpipe(rsrc_out) > -1)
      {
        if ((out = ::fdopen(rpipe(rsrc_out), "r")))
        {
            open_files |= pstdout;
        }
      }
      if (rpipe(rsrc_err) > -1)
      {
        if ((err = ::fdopen(rpipe(rsrc_err), "r")))
        {
            open_files |= pstderr;
        }
      }
      return open_files;
    }

  /**
   *  @warning This function exposes the internals of the stream buffer and
   *  should be used with caution.
   *
   *  @param  in   A FILE* that will refer to the process' stdin.
   *  @param  out  A FILE* that will refer to the process' stdout.
   *  @param  err  A FILE* that will refer to the process' stderr.
   *  @return A bitwise-or of zero or more of @c pstdin, @c pstdout, @c pstderr.
   *  @see    basic_pstreambuf::fopen()
   */
  template <typename C, typename T>
    inline std::size_t
    pstream_common<C,T>::fopen(FILE*& fin, FILE*& fout, FILE*& ferr)
    {
      return buf_.fopen(fin, fout, ferr);
    }

#endif // REDI_EVISCERATE_PSTREAMS


} // namespace redi

#endif  // REDI_PSTREAM_H_SEEN
#endif  // WIN32

/* basic_fun.h */
/* File parsing and basic geometry operations */

void PrintErrorAndQuit(const string sErrorString)
{
    cout << sErrorString << endl;
    exit(1);
}

template <typename T> inline T getmin(const T &a, const T &b)
{
    return b<a?b:a;
}

template <class A> void NewArray(A *** array, int Narray1, int Narray2)
{
    *array=new A* [Narray1];
    for(int i=0; i<Narray1; i++) *(*array+i)=new A [Narray2];
}

template <class A> void DeleteArray(A *** array, int Narray)
{
    for(int i=0; i<Narray; i++)
        if(*(*array+i)) delete [] *(*array+i);
    if(Narray) delete [] (*array);
    (*array)=NULL;
}

string AAmap(char A)
{
    if (A=='A') return "ALA";
    if (A=='B') return "ASX";
    if (A=='C') return "CYS";
    if (A=='D') return "ASP";
    if (A=='E') return "GLU";
    if (A=='F') return "PHE";
    if (A=='G') return "GLY";
    if (A=='H') return "HIS";
    if (A=='I') return "ILE";
    if (A=='K') return "LYS";
    if (A=='L') return "LEU";
    if (A=='M') return "MET";
    if (A=='N') return "ASN";
    if (A=='O') return "PYL";
    if (A=='P') return "PRO";
    if (A=='Q') return "GLN";
    if (A=='R') return "ARG";
    if (A=='S') return "SER";
    if (A=='T') return "THR";
    if (A=='U') return "SEC";
    if (A=='V') return "VAL";
    if (A=='W') return "TRP";    
    if (A=='Y') return "TYR";
    if (A=='Z') return "GLX";
    if ('a'<=A && A<='z') return "  "+string(1,char(toupper(A)));
    return "UNK";
}

char AAmap(const string &AA)
{
    if (AA.compare("ALA")==0 || AA.compare("DAL")==0) return 'A';
    if (AA.compare("ASX")==0) return 'B';
    if (AA.compare("CYS")==0 || AA.compare("DCY")==0) return 'C';
    if (AA.compare("ASP")==0 || AA.compare("DAS")==0) return 'D';
    if (AA.compare("GLU")==0 || AA.compare("DGL")==0) return 'E';
    if (AA.compare("PHE")==0 || AA.compare("DPN")==0) return 'F';
    if (AA.compare("GLY")==0) return 'G';
    if (AA.compare("HIS")==0 || AA.compare("DHI")==0) return 'H';
    if (AA.compare("ILE")==0 || AA.compare("DIL")==0) return 'I';
    if (AA.compare("LYS")==0 || AA.compare("DLY")==0) return 'K';
    if (AA.compare("LEU")==0 || AA.compare("DLE")==0) return 'L';
    if (AA.compare("MET")==0 || AA.compare("MED")==0 ||
        AA.compare("MSE")==0) return 'M';
    if (AA.compare("ASN")==0 || AA.compare("DSG")==0) return 'N';
    if (AA.compare("PYL")==0) return 'O';
    if (AA.compare("PRO")==0 || AA.compare("DPR")==0) return 'P';
    if (AA.compare("GLN")==0 || AA.compare("DGN")==0) return 'Q';
    if (AA.compare("ARG")==0 || AA.compare("DAR")==0) return 'R';
    if (AA.compare("SER")==0 || AA.compare("DSN")==0) return 'S';
    if (AA.compare("THR")==0 || AA.compare("DTH")==0) return 'T';
    if (AA.compare("SEC")==0) return 'U';
    if (AA.compare("VAL")==0 || AA.compare("DVA")==0) return 'V';
    if (AA.compare("TRP")==0 || AA.compare("DTR")==0) return 'W';    
    if (AA.compare("TYR")==0 || AA.compare("DTY")==0) return 'Y';
    if (AA.compare("GLX")==0) return 'Z';

    if (AA.compare(0,2," D")==0) return tolower(AA[2]);
    if (AA.compare(0,2,"  ")==0) return tolower(AA[2]);
    return 'X';
}

/* split a long string into vectors by whitespace 
 * line          - input string
 * line_vec      - output vector 
 * delimiter     - delimiter */
void split(const string &line, vector<string> &line_vec,
    const char delimiter=' ')
{
    bool within_word = false;
    for (size_t pos=0;pos<line.size();pos++)
    {
        if (line[pos]==delimiter)
        {
            within_word = false;
            continue;
        }
        if (!within_word)
        {
            within_word = true;
            line_vec.push_back("");
        }
        line_vec.back()+=line[pos];
    }
}

/* strip white space at the begining or end of string */
string Trim(const string &inputString)
{
    string result = inputString;
    int idxBegin = inputString.find_first_not_of(" \n\r\t");
    int idxEnd = inputString.find_last_not_of(" \n\r\t");
    if (idxBegin >= 0 && idxEnd >= 0)
        result = inputString.substr(idxBegin, idxEnd + 1 - idxBegin);
    return result;
}

size_t get_PDB_lines(const string filename,
    vector<vector<string> >&PDB_lines, vector<string> &chainID_list,
    vector<int> &mol_vec, const int ter_opt, const int infmt_opt,
    const string atom_opt, const int split_opt, const int het_opt)
{
    size_t i=0; // resi i.e. atom index
    string line;
    char chainID=0;
    string resi="";
    bool select_atom=false;
    size_t model_idx=0;
    vector<string> tmp_str_vec;
    
    int compress_type=0; // uncompressed file
    ifstream fin;
#ifndef REDI_PSTREAM_H_SEEN
    ifstream fin_gz;
#else
    redi::ipstream fin_gz; // if file is compressed
    if (filename.size()>=3 && 
        filename.substr(filename.size()-3,3)==".gz")
    {
        fin_gz.open("gunzip -c '"+filename+"'");
        compress_type=1;
    }
    else if (filename.size()>=4 && 
        filename.substr(filename.size()-4,4)==".bz2")
    {
        fin_gz.open("bzcat '"+filename+"'");
        compress_type=2;
    }
    else
#endif
    {
        if (filename=="-") compress_type=-1;
        else fin.open(filename.c_str());
    }

    if (infmt_opt==0||infmt_opt==-1) // PDB format
    {
        while ((compress_type==-1)?cin.good():(compress_type?fin_gz.good():fin.good()))
        {
            if  (compress_type==-1) getline(cin, line);
            else if (compress_type) getline(fin_gz, line);
            else                    getline(fin, line);
            if (infmt_opt==-1 && line.compare(0,5,"loop_")==0) // PDBx/mmCIF
                return get_PDB_lines(filename,PDB_lines,chainID_list,
                    mol_vec, ter_opt, 3, atom_opt, split_opt,het_opt);
            if (i > 0)
            {
                if      (ter_opt>=1 && line.compare(0,3,"END")==0) break;
                else if (ter_opt>=3 && line.compare(0,3,"TER")==0) break;
            }
            if (split_opt && line.compare(0,3,"END")==0) chainID=0;
            if (line.size()>=54 && (line[16]==' ' || line[16]=='A') && (
                (line.compare(0, 6, "ATOM  ")==0) || 
                (line.compare(0, 6, "HETATM")==0 && het_opt==1) ||
                (line.compare(0, 6, "HETATM")==0 && het_opt==2 && 
                 line.compare(17,3, "MSE")==0)))
            {
                if (atom_opt=="auto")
                {
                    if (line[17]==' ' && (line[18]=='D'||line[18]==' '))
                         select_atom=(line.compare(12,4," C3'")==0);
                    else select_atom=(line.compare(12,4," CA ")==0);
                }
                else if (atom_opt=="PC4'")
                {
                    if (line[17]==' ' && (line[18]=='D'||line[18]==' '))
                         select_atom=(line.compare(12,4," P  ")==0
                                  )||(line.compare(12,4," C4'")==0);
                    else select_atom=(line.compare(12,4," CA ")==0);
                }
                else     select_atom=(line.compare(12,4,atom_opt)==0);
                if (select_atom)
                {
                    if (!chainID)
                    {
                        chainID=line[21];
                        model_idx++;
                        stringstream i8_stream;
                        i=0;
                        if (split_opt==2) // split by chain
                        {
                            if (chainID==' ')
                            {
                                if (ter_opt>=1) i8_stream << ":_";
                                else i8_stream<<':'<<model_idx<<",_";
                            }
                            else
                            {
                                if (ter_opt>=1) i8_stream << ':' << chainID;
                                else i8_stream<<':'<<model_idx<<','<<chainID;
                            }
                            chainID_list.push_back(i8_stream.str());
                        }
                        else if (split_opt==1) // split by model
                        {
                            i8_stream << ':' << model_idx;
                            chainID_list.push_back(i8_stream.str());
                        }
                        PDB_lines.push_back(tmp_str_vec);
                        mol_vec.push_back(0);
                    }
                    else if (ter_opt>=2 && chainID!=line[21]) break;
                    if (split_opt==2 && chainID!=line[21])
                    {
                        chainID=line[21];
                        i=0;
                        stringstream i8_stream;
                        if (chainID==' ')
                        {
                            if (ter_opt>=1) i8_stream << ":_";
                            else i8_stream<<':'<<model_idx<<",_";
                        }
                        else
                        {
                            if (ter_opt>=1) i8_stream << ':' << chainID;
                            else i8_stream<<':'<<model_idx<<','<<chainID;
                        }
                        chainID_list.push_back(i8_stream.str());
                        PDB_lines.push_back(tmp_str_vec);
                        mol_vec.push_back(0);
                    }

                    if (resi==line.substr(22,5) && atom_opt!="PC4'")
                        cerr<<"Warning! Duplicated residue "<<resi<<endl;
                    resi=line.substr(22,5); // including insertion code

                    PDB_lines.back().push_back(line);
                    if (line[17]==' ' && (line[18]=='D'||line[18]==' ')) mol_vec.back()++;
                    else mol_vec.back()--;
                    i++;
                }
            }
        }
    }
    else if (infmt_opt==1) // SPICKER format
    {
        size_t L=0;
        float x,y,z;
        stringstream i8_stream;
        while ((compress_type==-1)?cin.good():(compress_type?fin_gz.good():fin.good()))
        {
            if  (compress_type==-1)
            {
                cin>>L>>x>>y>>z;
                getline(cin, line);
                if (!cin.good()) break;
            }
            else if (compress_type)
            {
                fin_gz>>L>>x>>y>>z;
                getline(fin_gz, line);
                if (!fin_gz.good()) break;
            }
            else
            {
                fin   >>L>>x>>y>>z;
                getline(fin, line);
                if (!fin.good()) break;
            }
            model_idx++;
            stringstream i8_stream;
            i8_stream << ':' << model_idx;
            chainID_list.push_back(i8_stream.str());
            PDB_lines.push_back(tmp_str_vec);
            mol_vec.push_back(0);
            for (i=0;i<L;i++)
            {
                if  (compress_type==-1) cin>>x>>y>>z;
                else if (compress_type) fin_gz>>x>>y>>z;
                else                    fin   >>x>>y>>z;
                i8_stream<<"ATOM   "<<setw(4)<<i+1<<"  CA  UNK  "<<setw(4)
                    <<i+1<<"    "<<setiosflags(ios::fixed)<<setprecision(3)
                    <<setw(8)<<x<<setw(8)<<y<<setw(8)<<z;
                line=i8_stream.str();
                i8_stream.str(string());
                PDB_lines.back().push_back(line);
            }
            if  (compress_type==-1) getline(cin, line);
            else if (compress_type) getline(fin_gz, line);
            else                    getline(fin, line);
        }
    }
    else if (infmt_opt==2) // xyz format
    {
        size_t L=0;
        stringstream i8_stream;
        while ((compress_type==-1)?cin.good():(compress_type?fin_gz.good():fin.good()))
        {
            if (compress_type==-1)  getline(cin, line);
            else if (compress_type) getline(fin_gz, line);
            else                    getline(fin, line);
            L=atoi(line.c_str());
            if (compress_type==-1)  getline(cin, line);
            else if (compress_type) getline(fin_gz, line);
            else                    getline(fin, line);
            for (i=0;i<line.size();i++)
                if (line[i]==' '||line[i]=='\t') break;
            if (!((compress_type==-1)?cin.good():(compress_type?fin_gz.good():fin.good()))) break;
            chainID_list.push_back(':'+line.substr(0,i));
            PDB_lines.push_back(tmp_str_vec);
            mol_vec.push_back(0);
            for (i=0;i<L;i++)
            {
                if (compress_type==-1)  getline(cin, line);
                else if (compress_type) getline(fin_gz, line);
                else                    getline(fin, line);
                i8_stream<<"ATOM   "<<setw(4)<<i+1<<"  CA  "
                    <<AAmap(line[0])<<"  "<<setw(4)<<i+1<<"    "
                    <<line.substr(2,8)<<line.substr(11,8)<<line.substr(20,8);
                line=i8_stream.str();
                i8_stream.str(string());
                PDB_lines.back().push_back(line);
                if (line[0]>='a' && line[0]<='z') mol_vec.back()++; // RNA
                else mol_vec.back()--;
            }
        }
    }
    else if (infmt_opt==3) // PDBx/mmCIF format
    {
        bool loop_ = false; // not reading following content
        map<string,int> _atom_site;
        int atom_site_pos;
        vector<string> line_vec;
        string alt_id=".";  // alternative location indicator
        string asym_id="."; // this is similar to chainID, except that
                            // chainID is char while asym_id is a string
                            // with possibly multiple char
        string prev_asym_id="";
        string AA="";       // residue name
        string atom="";
        string prev_resi="";
        string model_index=""; // the same as model_idx but type is string
        stringstream i8_stream;
        while ((compress_type==-1)?cin.good():(compress_type?fin_gz.good():fin.good()))
        {
            if (compress_type==-1)  getline(cin, line);
            else if (compress_type) getline(fin_gz, line);
            else                    getline(fin, line);
            if (line.size()==0) continue;
            if (loop_) loop_ = (line.size()>=2)?(line.compare(0,2,"# ")):(line.compare(0,1,"#"));
            if (!loop_)
            {
                if (line.compare(0,5,"loop_")) continue;
                while(1)
                {
                    if (compress_type==-1)
                    {
                        if (cin.good()) getline(cin, line);
                        else PrintErrorAndQuit("ERROR! Unexpected end of -");
                    }
                    else if (compress_type)
                    {
                        if (fin_gz.good()) getline(fin_gz, line);
                        else PrintErrorAndQuit("ERROR! Unexpected end of "+filename);
                    }
                    else
                    {
                        if (fin.good()) getline(fin, line);
                        else PrintErrorAndQuit("ERROR! Unexpected end of "+filename);
                    }
                    if (line.size()) break;
                }
                if (line.compare(0,11,"_atom_site.")) continue;

                loop_=true;
                _atom_site.clear();
                atom_site_pos=0;
                _atom_site[Trim(line.substr(11))]=atom_site_pos;

                while(1)
                {
                    if  (compress_type==-1) getline(cin, line);
                    else if (compress_type) getline(fin_gz, line);
                    else                    getline(fin, line);
                    if (line.size()==0) continue;
                    if (line.compare(0,11,"_atom_site.")) break;
                    _atom_site[Trim(line.substr(11))]=++atom_site_pos;
                }


                if (_atom_site.count("group_PDB")*
                    _atom_site.count("label_atom_id")*
                    _atom_site.count("label_comp_id")*
                   (_atom_site.count("auth_asym_id")+
                    _atom_site.count("label_asym_id"))*
                   (_atom_site.count("auth_seq_id")+
                    _atom_site.count("label_seq_id"))*
                    _atom_site.count("Cartn_x")*
                    _atom_site.count("Cartn_y")*
                    _atom_site.count("Cartn_z")==0)
                {
                    loop_ = false;
                    cerr<<"Warning! Missing one of the following _atom_site data items: group_PDB, label_atom_id, label_comp_id, auth_asym_id/label_asym_id, auth_seq_id/label_seq_id, Cartn_x, Cartn_y, Cartn_z"<<endl;
                    continue;
                }
            }

            line_vec.clear();
            split(line,line_vec);
            if ((line_vec[_atom_site["group_PDB"]]!="ATOM" &&
                 line_vec[_atom_site["group_PDB"]]!="HETATM") ||
                (line_vec[_atom_site["group_PDB"]]=="HETATM" &&
                 (het_opt==0 || 
                 (het_opt==2 && line_vec[_atom_site["label_comp_id"]]!="MSE")))
                ) continue;
            
            alt_id=".";
            if (_atom_site.count("label_alt_id")) // in 39.4 % of entries
                alt_id=line_vec[_atom_site["label_alt_id"]];
            if (alt_id!="." && alt_id!="A") continue;

            atom=line_vec[_atom_site["label_atom_id"]];
            if (atom[0]=='"') atom=atom.substr(1);
            if (atom.size() && atom[atom.size()-1]=='"')
                atom=atom.substr(0,atom.size()-1);
            if (atom.size()==0) continue;
            if      (atom.size()==1) atom=" "+atom+"  ";
            else if (atom.size()==2) atom=" "+atom+" "; // wrong for sidechain H
            else if (atom.size()==3) atom=" "+atom;
            else if (atom.size()>=5) continue;

            AA=line_vec[_atom_site["label_comp_id"]]; // residue name
            if      (AA.size()==1) AA="  "+AA;
            else if (AA.size()==2) AA=" " +AA;
            else if (AA.size()>=4) continue;

            if (atom_opt=="auto")
            {
                if (AA[0]==' ' && (AA[1]=='D'||AA[1]==' ')) // DNA || RNA
                     select_atom=(atom==" C3'");
                else select_atom=(atom==" CA ");
            }
            else if (atom_opt=="PC4'")
            {
                if (line[17]==' ' && (line[18]=='D'||line[18]==' '))
                     select_atom=(line.compare(12,4," P  ")==0
                              )||(line.compare(12,4," C4'")==0);
                else select_atom=(line.compare(12,4," CA ")==0);
            }
            else     select_atom=(atom==atom_opt);

            if (!select_atom) continue;

            if (_atom_site.count("auth_asym_id"))
                 asym_id=line_vec[_atom_site["auth_asym_id"]];
            else asym_id=line_vec[_atom_site["label_asym_id"]];
            if (asym_id==".") asym_id=" ";
            
            if (_atom_site.count("pdbx_PDB_model_num") && 
                model_index!=line_vec[_atom_site["pdbx_PDB_model_num"]])
            {
                model_index=line_vec[_atom_site["pdbx_PDB_model_num"]];
                if (PDB_lines.size() && ter_opt>=1) break;
                if (PDB_lines.size()==0 || split_opt>=1)
                {
                    PDB_lines.push_back(tmp_str_vec);
                    mol_vec.push_back(0);
                    prev_asym_id=asym_id;

                    if (split_opt==1 && ter_opt==0) chainID_list.push_back(
                        ':'+model_index);
                    else if (split_opt==2 && ter_opt==0)
                        chainID_list.push_back(':'+model_index+','+asym_id);
                    else //if (split_opt==2 && ter_opt==1)
                        chainID_list.push_back(':'+asym_id);
                    //else
                        //chainID_list.push_back("");
                }
            }

            if (prev_asym_id!=asym_id)
            {
                if (prev_asym_id!="" && ter_opt>=2) break;
                if (split_opt>=2)
                {
                    PDB_lines.push_back(tmp_str_vec);
                    mol_vec.push_back(0);

                    if (split_opt==1 && ter_opt==0) chainID_list.push_back(
                        ':'+model_index);
                    else if (split_opt==2 && ter_opt==0)
                        chainID_list.push_back(':'+model_index+','+asym_id);
                    else //if (split_opt==2 && ter_opt==1)
                        chainID_list.push_back(':'+asym_id);
                    //else
                        //chainID_list.push_back("");
                }
            }
            if (prev_asym_id!=asym_id) prev_asym_id=asym_id;

            if (AA[0]==' ' && (AA[1]=='D'||AA[1]==' ')) mol_vec.back()++;
            else mol_vec.back()--;

            if (_atom_site.count("auth_seq_id"))
                 resi=line_vec[_atom_site["auth_seq_id"]];
            else resi=line_vec[_atom_site["label_seq_id"]];
            if (_atom_site.count("pdbx_PDB_ins_code") && 
                line_vec[_atom_site["pdbx_PDB_ins_code"]]!="?")
                resi+=line_vec[_atom_site["pdbx_PDB_ins_code"]][0];
            else resi+=" ";

            if (prev_resi==resi && atom_opt!="PC4'")
                cerr<<"Warning! Duplicated residue "<<resi<<endl;
            prev_resi=resi;

            i++;
            i8_stream<<"ATOM  "
                <<setw(5)<<i<<" "<<atom<<" "<<AA<<" "<<asym_id[0]
                <<setw(5)<<resi.substr(0,5)<<"   "
                <<setw(8)<<line_vec[_atom_site["Cartn_x"]].substr(0,8)
                <<setw(8)<<line_vec[_atom_site["Cartn_y"]].substr(0,8)
                <<setw(8)<<line_vec[_atom_site["Cartn_z"]].substr(0,8);
            PDB_lines.back().push_back(i8_stream.str());
            i8_stream.str(string());
        }
        _atom_site.clear();
        line_vec.clear();
        alt_id.clear();
        asym_id.clear();
        AA.clear();
    }

    if      (compress_type>=1) fin_gz.close();
    else if (compress_type==0) fin.close();
    line.clear();
    if (!split_opt) chainID_list.push_back("");
    return PDB_lines.size();
}

/* read fasta file from filename. sequence is stored into FASTA_lines
 * while sequence name is stored into chainID_list.
 * if ter_opt >=1, only read the first sequence.
 * if ter_opt ==0, read all sequences.
 * if split_opt >=1 and ter_opt ==0, each sequence is a separate entry.
 * if split_opt ==0 and ter_opt ==0, all sequences are combined into one */
size_t get_FASTA_lines(const string filename,
    vector<vector<string> >&FASTA_lines, vector<string> &chainID_list,
    vector<int> &mol_vec, const int ter_opt=3, const int split_opt=0)
{
    string line;
    vector<string> tmp_str_vec;
    size_t l;
    
    int compress_type=0; // uncompressed file
    ifstream fin;
#ifndef REDI_PSTREAM_H_SEEN
    ifstream fin_gz;
#else
    redi::ipstream fin_gz; // if file is compressed
    if (filename.size()>=3 && 
        filename.substr(filename.size()-3,3)==".gz")
    {
        fin_gz.open("gunzip -c '"+filename+"'");
        compress_type=1;
    }
    else if (filename.size()>=4 && 
        filename.substr(filename.size()-4,4)==".bz2")
    {
        fin_gz.open("bzcat '"+filename+"'");
        compress_type=2;
    }
    else 
#endif
    {
        if (filename=="-") compress_type=-1;
        else fin.open(filename.c_str());
    }

    while ((compress_type==-1)?cin.good():(compress_type?fin_gz.good():fin.good()))
    {
        if  (compress_type==-1) getline(cin, line);
        else if (compress_type) getline(fin_gz, line);
        else                    getline(fin, line);

        if (line.size()==0 || line[0]=='#') continue;

        if (line[0]=='>')
        {
            if (FASTA_lines.size())
            {
                if (ter_opt) break;
                if (split_opt==0) continue;
            }
            FASTA_lines.push_back(tmp_str_vec);
            FASTA_lines.back().push_back("");
            mol_vec.push_back(0);
            if (ter_opt==0 && split_opt)
            {
                line[0]=':';
                chainID_list.push_back(line);
            }
            else chainID_list.push_back("");
        }
        else
        {
            FASTA_lines.back()[0]+=line;
            for (l=0;l<line.size();l++) mol_vec.back()+=
                ('a'<=line[l] && line[l]<='z')-('A'<=line[l] && line[l]<='Z');
        }
    }

    line.clear();
    if      (compress_type>=1) fin_gz.close();
    else if (compress_type==0) fin.close();
    return FASTA_lines.size();
}

int read_PDB(const vector<string> &PDB_lines, double **a, char *seq,
    vector<string> &resi_vec, const int read_resi)
{
    size_t i;
    for (i=0;i<PDB_lines.size();i++)
    {
        a[i][0] = atof(PDB_lines[i].substr(30, 8).c_str());
        a[i][1] = atof(PDB_lines[i].substr(38, 8).c_str());
        a[i][2] = atof(PDB_lines[i].substr(46, 8).c_str());
        seq[i]  = AAmap(PDB_lines[i].substr(17, 3));

        if (read_resi>=2) resi_vec.push_back(PDB_lines[i].substr(22,5)+
                                             PDB_lines[i][21]);
        if (read_resi==1) resi_vec.push_back(PDB_lines[i].substr(22,5));
    }
    seq[i]='\0'; 
    return i;
}

double dist(double x[3], double y[3])
{
    double d1=x[0]-y[0];
    double d2=x[1]-y[1];
    double d3=x[2]-y[2];
 
    return (d1*d1 + d2*d2 + d3*d3);
}

double dot(double *a, double *b)
{
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

void transform(double t[3], double u[3][3], double *x, double *x1)
{
    x1[0]=t[0]+dot(&u[0][0], x);
    x1[1]=t[1]+dot(&u[1][0], x);
    x1[2]=t[2]+dot(&u[2][0], x);
}

void do_rotation(double **x, double **x1, int len, double t[3], double u[3][3])
{
    for(int i=0; i<len; i++)
    {
        transform(t, u, &x[i][0], &x1[i][0]);
    }    
}

/* read user specified pairwise alignment from 'fname_lign' to 'sequence'.
 * This function should only be called by main function, as it will
 * terminate a program if wrong alignment is given */
void read_user_alignment(vector<string>&sequence, const string &fname_lign,
    const int i_opt)
{
    if (fname_lign == "")
        PrintErrorAndQuit("Please provide a file name for option -i!");
    // open alignment file
    int n_p = 0;// number of structures in alignment file
    string line;
    
    ifstream fileIn(fname_lign.c_str());
    if (fileIn.is_open())
    {
        while (fileIn.good())
        {
            getline(fileIn, line);
            if (line.compare(0, 1, ">") == 0)// Flag for a new structure
            {
                if (n_p >= 2) break;
                sequence.push_back("");
                n_p++;
            }
            else if (n_p > 0 && line!="") sequence.back()+=line;
        }
        fileIn.close();
    }
    else PrintErrorAndQuit("ERROR! Alignment file does not exist.");
    
    if (n_p < 2)
        PrintErrorAndQuit("ERROR: Fasta format is wrong, two proteins should be included.");
    if (sequence[0].size() != sequence[1].size())
        PrintErrorAndQuit("ERROR! FASTA file is wrong. The length in alignment should be equal for the two aligned proteins.");
    if (i_opt==3)
    {
        int aligned_resNum=0;
        for (size_t i=0;i<sequence[0].size();i++)
            aligned_resNum+=(sequence[0][i]!='-' && sequence[1][i]!='-');
        if (aligned_resNum<3)
            PrintErrorAndQuit("ERROR! Superposition is undefined for <3 aligned residues.");
    }
    line.clear();
    return;
}

/* read list of entries from 'name' to 'chain_list'.
 * dir_opt is the folder name (prefix).
 * suffix_opt is the file name extension (suffix_opt).
 * This function should only be called by main function, as it will
 * terminate a program if wrong alignment is given */
void file2chainlist(vector<string>&chain_list, const string &name,
    const string &dir_opt, const string &suffix_opt)
{
    ifstream fp(name.c_str());
    if (! fp.is_open())
        PrintErrorAndQuit(("Can not open file: "+name+'\n').c_str());
    string line;
    while (fp.good())
    {
        getline(fp, line);
        if (! line.size()) continue;
        chain_list.push_back(dir_opt+Trim(line)+suffix_opt);
    }
    fp.close();
    line.clear();
}

/* BLOSUM.h */
/* This matrix contains two scoring matrices: 
 * [1] BLOSUM62 for protein is defined for upper case letters:
 *     ABCDEFGHIKLMNOPQRSTVWXYZ* excluding J
 *     The original BLOSUM does not have O (PYL) and U (SEC).
 *     In this matrix, OU values are copied from K and C, respectively.
 * [2] BLASTN for RNA/DNA is defined for lower case letters:
 *     acgtu where matching (including t vs u) is 2 and mismatching is -3 */

const int BLOSUM[128][128]={
//0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 51 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
//0                   \a \b \t \n \v \f \t                                                       ' ' |  "  #  $  %  &  '  (  )  *  +  ,  -  .  /  0  1  2  3  4  5  6  7  8  9  :  ;  <  =  >  ?  @  A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z  [  \  ]  ^  _  `  a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r  s  t  u  v  w  x  y  z  {  |  }  ~ DEL
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//0   '\0'
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//1   SOH
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//2   STX
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//3   ETX
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//4   EOT
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//5   ENQ
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//6   ACK
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//7   '\a'
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//8   '\b'
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//9   '\t'
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//10  '\n'
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//11  '\v'
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//12  '\f'
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//13  '\r'
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//14  SO
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//15  SI
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//16  DLE
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//17  DC1
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//18  DC2
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//19  DC3
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//20  DC4
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//21  NAK
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//22  SYN
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//23  ETB
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//24  CAN
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//25  EM
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//26  SUB
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//27  ESC
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//28  FS
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//29  GS
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//30  RS
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//31  US
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//32  ' '
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//33  !    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//34  "    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//35  #    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//36  $    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//37  %    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//38  &    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//39  '    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//40  (    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//41  )    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4,-4,-4,-4,-4,-4,-4,-4,-4, 0,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//42  *
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//43  +    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//44  ,    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//45  -    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//46  .    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//47  /    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//48  0    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//49  1    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//50  2    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//51  3    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//52  4    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//53  5    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//54  6    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//55  7    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//56  8    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//57  9    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//58  :    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//59  ;    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//60  <    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//61  =    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//62  >    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//63  ?    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//64  @    
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,-2, 0,-2,-1,-2, 0,-2,-1, 0,-1,-1,-1,-2,-1,-1,-1,-1, 1, 0, 0, 0,-3, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//65  A
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 4,-3, 4, 1,-3,-1, 0,-3, 0, 0,-4,-3, 3, 0,-2, 0,-1, 0,-1,-3,-3,-4,-1,-3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//66  B
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 9,-3,-4,-2,-3,-3,-1, 0,-3,-1,-1,-3,-3,-3,-3,-3,-1,-1, 9,-1,-2,-2,-2,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//67  C
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 4,-3, 6, 2,-3,-1,-1,-3, 0,-1,-4,-3, 1,-1,-1, 0,-2, 0,-1,-3,-3,-4,-1,-3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//68  D
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1,-4, 2, 5,-3,-2, 0,-3, 0, 1,-3,-2, 0, 1,-1, 2, 0, 0,-1,-4,-2,-3,-1,-2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//69  E
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2,-3,-2,-3,-3, 6,-3,-1, 0, 0,-3, 0, 0,-3,-3,-4,-3,-3,-2,-2,-2,-1, 1,-1, 3,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//70  F
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-3,-1,-2,-3, 6,-2,-4, 0,-2,-4,-3, 0,-2,-2,-2,-2, 0,-2,-3,-3,-2,-1,-3,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//71  G
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-3,-1, 0,-1,-2, 8,-3, 0,-1,-3,-2, 1,-1,-2, 0, 0,-1,-2,-3,-3,-2,-1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//72  H
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-3,-1,-3,-3, 0,-4,-3, 4, 0,-3, 2, 1,-3,-3,-3,-3,-3,-2,-1,-1, 3,-3,-1,-1,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//73  I
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//74  J
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,-3,-1, 1,-3,-2,-1,-3, 0, 5,-2,-1, 0, 5,-1, 1, 2, 0,-1,-3,-2,-3,-1,-2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//75  K
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-4,-1,-4,-3, 0,-4,-3, 2, 0,-2, 4, 2,-3,-2,-3,-2,-2,-2,-1,-1, 1,-2,-1,-1,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//76  L
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-3,-1,-3,-2, 0,-3,-2, 1, 0,-1, 2, 5,-2,-1,-2, 0,-1,-1,-1,-1, 1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//77  M
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 3,-3, 1, 0,-3, 0, 1,-3, 0, 0,-3,-2, 6, 0,-2, 0, 0, 1, 0,-3,-3,-4,-1,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//78  N
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,-3,-1, 1,-3,-2,-1,-3, 0, 5,-2,-1, 0, 5,-1, 1, 2, 0,-1,-3,-2,-3,-1,-2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//79  O
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-2,-3,-1,-1,-4,-2,-2,-3, 0,-1,-3,-2,-2,-1, 7,-1,-2,-1,-1,-3,-2,-4,-2,-3,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//80  P
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,-3, 0, 2,-3,-2, 0,-3, 0, 1,-2, 0, 0, 1,-1, 5, 1, 0,-1,-3,-2,-2,-1,-1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//81  Q
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1,-3,-2, 0,-3,-2, 0,-3, 0, 2,-2,-1, 0, 2,-2, 1, 5,-1,-1,-3,-3,-3,-1,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//82  R
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0,-2, 0,-1,-2, 0, 0,-2,-1, 1, 0,-1, 0,-1, 4, 1,-1,-2,-3, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//83  S
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1,-1,-1,-2,-2,-2,-1, 0,-1,-1,-1, 0,-1,-1,-1,-1, 1, 5,-1, 0,-2, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//84  T
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 9,-3,-4,-2,-3,-3,-1, 0,-3,-1,-1,-3,-3,-3,-3,-3,-1,-1, 9,-1,-2,-2,-2,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//85  U
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3,-1,-3,-2,-1,-3,-3, 3, 0,-2, 1, 1,-3,-2,-2,-2,-3,-2, 0,-1, 4,-3,-1,-1,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//86  V
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3,-4,-2,-4,-3, 1,-2,-2,-3, 0,-3,-2,-1,-4,-3,-4,-2,-3,-3,-2,-2,-3,11,-2, 2,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//87  W
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-2,-1,-1,-1,-1,-1,-1, 0,-1,-1,-1,-1,-1,-2,-1,-1, 0, 0,-2,-1,-2,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//88  X
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2,-3,-2,-3,-2, 3,-3, 2,-1, 0,-2,-1,-1,-2,-2,-3,-1,-2,-2,-2,-2,-1, 2,-1, 7,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//89  Y
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1,-3, 1, 4,-3,-2, 0,-3, 0, 1,-3,-1, 0, 1,-1, 3, 0, 0,-1,-3,-2,-3,-1,-2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//90  Z
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//91  [
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//92  '\'
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//93  ]
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//94  ^
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//95  _
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//96  `
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-3, 0, 0, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//97  a
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//98  b
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 2, 0, 0, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//99  c
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//100 d
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//101 e
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//102 f
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0,-3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//103 g
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//104 h
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//105 i
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//106 j
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//107 k
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//108 l
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//109 m
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//110 n
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//111 o
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//112 p
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//113 q
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//114 r
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//115 s
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0,-3, 0, 0, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//116 t
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0,-3, 0, 0, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//117 u
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//118 v
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//119 w
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//120 x
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//121 y
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//122 z
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//123 {
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//124 |
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//125 }
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//126 ~
{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},//127 DEL
};

#define MAX(A,B) ((A)>(B)?(A):(B))

const int gapopen_blosum62=-11;
const int gapext_blosum62=-1;

const int gapopen_blastn=-15; //-5;
const int gapext_blastn =-4;  //-2;

/* initialize matrix in gotoh algorithm */
void init_gotoh_mat(int **S, int **JumpH, int **JumpV, int **P,
    int **H, int **V, const int xlen, const int ylen, const int gapopen,
    const int gapext, const int glocal=0, const int alt_init=1)
{
    // fill first row/colum of JumpH,jumpV and path matrix P
    int i,j;
    for (i=0;i<xlen+1;i++)
        for (j=0;j<ylen+1;j++)
            H[i][j]=V[i][j]=P[i][j]=JumpH[i][j]=JumpV[i][j]=0;
    for (i=0;i<xlen+1;i++)
    {
        if (glocal<2) P[i][0]=4; // -
        JumpV[i][0]=i;
    }
    for (j=0;j<ylen+1;j++)
    {
        if (glocal<1) P[0][j]=2; // |
        JumpH[0][j]=j;
    }
    if (glocal<2) for (i=1;i<xlen+1;i++) S[i][0]=gapopen+gapext*(i-1);
    if (glocal<1) for (j=1;j<ylen+1;j++) S[0][j]=gapopen+gapext*(j-1);
    if (alt_init==0)
    {
        for (i=1;i<xlen+1;i++) H[i][0]=gapopen+gapext*(i-1);
        for (j=1;j<ylen+1;j++) V[0][j]=gapopen+gapext*(j-1);
    }
    else
    {
        if (glocal<2) for (i=1;i<xlen+1;i++) V[i][0]=gapopen+gapext*(i-1);
        if (glocal<1) for (j=1;j<ylen+1;j++) H[0][j]=gapopen+gapext*(j-1);
        for (i=0;i<xlen+1;i++) H[i][0]=-99999; // INT_MIN cause bug on ubuntu
        for (j=0;j<ylen+1;j++) V[0][j]=-99999; // INT_MIN;
    }
}

/* locate the cell with highest alignment score. reset path after
 * the cell to zero */
void find_highest_align_score( int **S, int **P,
    int &aln_score, const int xlen,const int ylen)
{
    // locate the cell with highest alignment score
    int max_aln_i=xlen;
    int max_aln_j=ylen;
    int i,j;
    for (i=0;i<xlen+1;i++)
    {
        for (j=0;j<ylen+1;j++)
        {
            if (S[i][j]>=aln_score)
            {
                max_aln_i=i;
                max_aln_j=j;
                aln_score=S[i][j];
            }
        }
    }

    // reset all path after [max_aln_i][max_aln_j]
    for (i=max_aln_i+1;i<xlen+1;i++) for (j=0;j<ylen+1;j++) P[i][j]=0;
    for (i=0;i<xlen+1;i++) for (j=max_aln_j+1;j<ylen+1;j++) P[i][j]=0;
}

/* calculate dynamic programming matrix using gotoh algorithm
 * S     - cumulative scorefor each cell
 * P     - string representation for path
 *         0 :   uninitialized, for gaps at N- & C- termini when glocal>0
 *         1 : \ match-mismatch
 *         2 : | vertical gap (insertion)
 *         4 : - horizontal gap (deletion)
 * JumpH - horizontal long gap number.
 * JumpV - vertical long gap number.
 * all matrices are in the size of [len(seqx)+1]*[len(seqy)+1]
 *
 * glocal - global or local alignment
 *         0 : global alignment (Needleman-Wunsch dynamic programming)
 *         1 : glocal-query alignment
 *         2 : glocal-both alignment
 *         3 : local alignment (Smith-Waterman dynamic programming)
 *
 * alt_init - whether to adopt alternative matrix initialization
 *         1 : use wei zheng's matrix initialization
 *         0 : use yang zhang's matrix initialization, does NOT work
 *             for glocal alignment
 */
int calculate_score_gotoh(const int xlen,const int ylen, int **S,
    int** JumpH, int** JumpV, int **P, const int gapopen,const int gapext,
    const int glocal=0, const int alt_init=1)
{
    int **H;
    int **V;
    NewArray(&H,xlen+1,ylen+1); // penalty score for horizontal long gap
    NewArray(&V,xlen+1,ylen+1); // penalty score for vertical long gap
    
    // fill first row/colum of JumpH,jumpV and path matrix P
    int i,j;
    init_gotoh_mat(S, JumpH, JumpV, P, H, V, xlen, ylen,
        gapopen, gapext, glocal, alt_init);

    // fill S and P
    int diag_score,left_score,up_score;
    for (i=1;i<xlen+1;i++)
    {
        for (j=1;j<ylen+1;j++)
        {
            // penalty of consective deletion
            if (glocal<1 || i<xlen || glocal>=3)
            {
                H[i][j]=MAX(S[i][j-1]+gapopen,H[i][j-1]+gapext);
                JumpH[i][j]=(H[i][j]==H[i][j-1]+gapext)?(JumpH[i][j-1]+1):1;
            }
            else
            {
                H[i][j]=MAX(S[i][j-1],H[i][j-1]);
                JumpH[i][j]=(H[i][j]==H[i][j-1])?(JumpH[i][j-1]+1):1;
            }
            // penalty of consective insertion
            if (glocal<2 || j<ylen || glocal>=3)
            {
                V[i][j]=MAX(S[i-1][j]+gapopen,V[i-1][j]+gapext);
                JumpV[i][j]=(V[i][j]==V[i-1][j]+gapext)?(JumpV[i-1][j]+1):1;
            }
            else
            {
                V[i][j]=MAX(S[i-1][j],V[i-1][j]);
                JumpV[i][j]=(V[i][j]==V[i-1][j])?(JumpV[i-1][j]+1):1;
            }

            diag_score=S[i-1][j-1]+S[i][j]; // match-mismatch '\'
            left_score=H[i][j];     // deletion       '-'
            up_score  =V[i][j];     // insertion      '|'

            if (diag_score>=left_score && diag_score>=up_score)
            {
                S[i][j]=diag_score;
                P[i][j]+=1;
            }
            if (up_score>=diag_score && up_score>=left_score)
            {
                S[i][j]=up_score;
                P[i][j]+=2;
            }
            if (left_score>=diag_score && left_score>=up_score)
            {
                S[i][j]=left_score;
                P[i][j]+=4;
            }
            if (glocal>=3 && S[i][j]<0)
            {
                S[i][j]=0;
                P[i][j]=0;
                H[i][j]=0;
                V[i][j]=0;
                JumpH[i][j]=0;
                JumpV[i][j]=0;
            }
        }
    }
    int aln_score=S[xlen][ylen];

    // re-fill first row/column of path matrix P for back-tracing
    for (i=1;i<xlen+1;i++) if (glocal<3 || P[i][0]>0) P[i][0]=2; // |
    for (j=1;j<ylen+1;j++) if (glocal<3 || P[0][j]>0) P[0][j]=4; // -

    // calculate alignment score and alignment path for swalign
    if (glocal>=3)
        find_highest_align_score(S,P,aln_score,xlen,ylen);

    // release memory
    DeleteArray(&H,xlen+1);
    DeleteArray(&V,xlen+1);
    return aln_score; // final alignment score
}

/* trace back dynamic programming path to diciper pairwise alignment */
void trace_back_gotoh(const char *seqx, const char *seqy,
    int ** JumpH, int ** JumpV, int ** P, string& seqxA, string& seqyA,
    const int xlen, const int ylen, int *invmap, const int invmap_only=1)
{
    int i,j;
    int gaplen,p;
    char *buf=NULL;

    if (invmap_only) for (j = 0; j < ylen; j++) invmap[j] = -1;
    if (invmap_only!=1) buf=new char [MAX(xlen,ylen)+1];

    i=xlen;
    j=ylen;
    while(i+j)
    {
        gaplen=0;
        if (P[i][j]>=4)
        {
            gaplen=JumpH[i][j];
            j-=gaplen;
            if (invmap_only==1) continue;
            strncpy(buf,seqy+j,gaplen);
            buf[gaplen]=0;
            seqyA=buf+seqyA;

            for (p=0;p<gaplen;p++) buf[p]='-';
            seqxA=buf+seqxA;
        }
        else if (P[i][j] % 4 >= 2)
        {
            gaplen=JumpV[i][j];
            i-=gaplen;
            if (invmap_only==1) continue;
            strncpy(buf,seqx+i,gaplen);
            buf[gaplen]=0;
            seqxA=buf+seqxA;

            for (p=0;p<gaplen;p++) buf[p]='-';
            seqyA=buf+seqyA;
        }
        else
        {
            if (i==0 && j!=0) // only in glocal alignment
            {
                strncpy(buf,seqy,j);
                buf[j]=0;
                seqyA=buf+seqyA;
                for (p=0;p<j;p++) buf[p]='-';
                seqxA=buf+seqxA;
                break;
            }
            if (i!=0 && j==0) // only in glocal alignment
            {
                strncpy(buf,seqx,i);
                buf[i]=0;
                seqxA=buf+seqxA;
                for (p=0;p<i;p++) buf[p]='-';
                seqyA=buf+seqyA;
                break;
            }
            i--;
            j--;
            if (invmap_only) invmap[j]=i;
            if (invmap_only!=1)
            {
                seqxA=seqx[i]+seqxA;
                seqyA=seqy[j]+seqyA;
            }
        }
    }
    delete [] buf;
}


/* trace back Smith-Waterman dynamic programming path to diciper 
 * pairwise local alignment */
void trace_back_sw(const char *seqx, const char *seqy,
    int **JumpH, int **JumpV, int **P, string& seqxA, string& seqyA,
    const int xlen, const int ylen, int *invmap, const int invmap_only=1)
{
    int i;
    int j;
    int gaplen,p;
    bool found_start_cell=false; // find the first non-zero cell in P
    char *buf=NULL;

    if (invmap_only) for (j = 0; j < ylen; j++) invmap[j] = -1;
    if (invmap_only!=1) buf=new char [MAX(xlen,ylen)+1];

    i=xlen;
    j=ylen;
    for (i=xlen;i>=0;i--)
    {
        for (j=ylen;j>=0;j--)
        {
            if (P[i][j]!=0)
            {
                found_start_cell=true;
                break;
            }
        }
        if (found_start_cell) break;
    }

    /* copy C terminal sequence */
    if (invmap_only!=1)
    {
        for (p=0;p<ylen-j;p++) buf[p]='-';
        buf[ylen-j]=0;
        seqxA=buf;
        strncpy(buf,seqx+i,xlen-i);
        buf[xlen-i]=0;
        seqxA+=buf;

        strncpy(buf,seqy+j,ylen-j);
        buf[ylen-j]=0;
        seqyA+=buf;
        for (p=0;p<xlen-i;p++) buf[p]='-';
        buf[xlen-i]=0;
        seqyA+=buf;
    }

    if (i<0||j<0)
    {
        delete [] buf;
        return;
    }

    /* traceback aligned sequences */
    while(P[i][j]!=0)
    {
        gaplen=0;
        if (P[i][j]>=4)
        {
            gaplen=JumpH[i][j];
            j-=gaplen;
            if (invmap_only==1) continue;
            strncpy(buf,seqy+j,gaplen);
            buf[gaplen]=0;
            seqyA=buf+seqyA;

            for (p=0;p<gaplen;p++) buf[p]='-';
            seqxA=buf+seqxA;
        }
        else if (P[i][j] % 4 >= 2)
        {
            gaplen=JumpV[i][j];
            i-=gaplen;
            if (invmap_only==1) continue;
            strncpy(buf,seqx+i,gaplen);
            buf[gaplen]=0;
            seqxA=buf+seqxA;

            for (p=0;p<gaplen;p++) buf[p]='-';
            seqyA=buf+seqyA;
        }
        else
        {
            i--;
            j--;
            if (invmap_only) invmap[j]=i;
            if (invmap_only!=1)
            {
                seqxA=seqx[i]+seqxA;
                seqyA=seqy[j]+seqyA;
            }
        }
    }
    /* copy N terminal sequence */
    if (invmap_only!=1)
    {
        for (p=0;p<j;p++) buf[p]='-';
        strncpy(buf+j,seqx,i);
        buf[i+j]=0;
        seqxA=buf+seqxA;

        strncpy(buf,seqy,j);
        for (p=j;p<j+i;p++) buf[p]='-';
        buf[i+j]=0;
        seqyA=buf+seqyA;
    }
    delete [] buf;
}

/* entry function for NWalign
 * invmap_only - whether to return seqxA and seqyA or to return invmap
 *               0: only return seqxA and seqyA
 *               1: only return invmap
 *               2: return seqxA, seqyA and invmap */
int NWalign_main(const char *seqx, const char *seqy, const int xlen,
    const int ylen, string & seqxA, string & seqyA, const int mol_type,
    int *invmap, const int invmap_only=0, const int glocal=0)
{
    int **JumpH;
    int **JumpV;
    int **P;
    int **S;
    NewArray(&JumpH,xlen+1,ylen+1);
    NewArray(&JumpV,xlen+1,ylen+1);
    NewArray(&P,xlen+1,ylen+1);
    NewArray(&S,xlen+1,ylen+1);
    
    int aln_score;
    int gapopen=gapopen_blosum62;
    int gapext =gapext_blosum62;
    int i,j;
    if (mol_type>0) // RNA or DNA
    {
        gapopen=gapopen_blastn;
        gapext =gapext_blastn;
        if (glocal==3)
        {
            gapopen=-5;
            gapext =-2;
        }
    }

    for (i=0;i<xlen+1;i++)
    {
        for (j=0;j<ylen+1;j++)
        {
            if (i*j==0) S[i][j]=0;
            else S[i][j]=BLOSUM[seqx[i-1]][seqy[j-1]];
        }
    }

    aln_score=calculate_score_gotoh(xlen, ylen, S, JumpH, JumpV, P,
        gapopen, gapext, glocal);

    seqxA.clear();
    seqyA.clear();

    if (glocal<3) trace_back_gotoh(seqx, seqy, JumpH, JumpV, P,
            seqxA, seqyA, xlen, ylen, invmap, invmap_only);
    else trace_back_sw(seqx, seqy, JumpH, JumpV, P, seqxA, seqyA,
            xlen, ylen, invmap, invmap_only);

    DeleteArray(&JumpH, xlen+1);
    DeleteArray(&JumpV, xlen+1);
    DeleteArray(&P, xlen+1);
    DeleteArray(&S, xlen+1);
    return aln_score; // aligment score
}

/* extract pairwise sequence alignment from residue index vectors,
 * assuming that "sequence" contains two empty strings.
 * return length of alignment, including gap. */
int extract_aln_from_resi(vector<string> &sequence, char *seqx, char *seqy,
    const vector<string> resi_vec1, const vector<string> resi_vec2,
    const int byresi_opt)
{
    sequence.clear();
    sequence.push_back("");
    sequence.push_back("");

    int i1=0; // positions in resi_vec1
    int i2=0; // positions in resi_vec2
    int xlen=resi_vec1.size();
    int ylen=resi_vec2.size();
    if (byresi_opt==4 || byresi_opt==5) // global or glocal sequence alignment
    {
        int *invmap;
        int glocal=0;
        if (byresi_opt==5) glocal=2;
        int mol_type=0;
        for (i1=0;i1<xlen;i1++)
            if ('a'<seqx[i1] && seqx[i1]<'z') mol_type++;
            else mol_type--;
        for (i2=0;i2<ylen;i2++)
            if ('a'<seqx[i2] && seqx[i2]<'z') mol_type++;
            else mol_type--;
        NWalign_main(seqx, seqy, xlen, ylen, sequence[0],sequence[1],
            mol_type, invmap, 0, glocal);
    }
    map<string,string> chainID_map1;
    map<string,string> chainID_map2;
    if (byresi_opt==3)
    {
        vector<string> chainID_vec;
        string chainID;
        stringstream ss;
        int i;
        for (i=0;i<xlen;i++)
        {
            chainID=resi_vec1[i].substr(5);
            if (!chainID_vec.size()|| chainID_vec.back()!=chainID)
            {
                chainID_vec.push_back(chainID);
                ss<<chainID_vec.size();
                chainID_map1[chainID]=ss.str();
                ss.str("");
            }
        }
        chainID_vec.clear();
        for (i=0;i<ylen;i++)
        {
            chainID=resi_vec2[i].substr(5);
            if (!chainID_vec.size()|| chainID_vec.back()!=chainID)
            {
                chainID_vec.push_back(chainID);
                ss<<chainID_vec.size();
                chainID_map2[chainID]=ss.str();
                ss.str("");
            }
        }
        vector<string>().swap(chainID_vec);
    }
    string chainID1="";
    string chainID2="";
    string chainID1_prev="";
    string chainID2_prev="";
    while(i1<xlen && i2<ylen)
    {
        if (byresi_opt==2)
        {
            chainID1=resi_vec1[i1].substr(5);
            chainID2=resi_vec2[i2].substr(5);
        }
        else if (byresi_opt==3)
        {
            chainID1=chainID_map1[resi_vec1[i1].substr(5)];
            chainID2=chainID_map2[resi_vec2[i2].substr(5)];
        }

        if (chainID1==chainID2)
        {
            if (atoi(resi_vec1[i1].substr(0,4).c_str())<
                atoi(resi_vec2[i2].substr(0,4).c_str()))
            {
                sequence[0]+=seqx[i1++];
                sequence[1]+='-';
            }
            else if (atoi(resi_vec1[i1].substr(0,4).c_str())>
                     atoi(resi_vec2[i2].substr(0,4).c_str()))
            {
                sequence[0]+='-';
                sequence[1]+=seqy[i2++];
            }
            else
            {
                sequence[0]+=seqx[i1++];
                sequence[1]+=seqy[i2++];
            }
            chainID1_prev=chainID1;
            chainID2_prev=chainID2;
        }
        else
        {
            if (chainID1_prev==chainID1 && chainID2_prev!=chainID2)
            {
                sequence[0]+=seqx[i1++];
                sequence[1]+='-';
                chainID1_prev=chainID1;
            }
            else if (chainID1_prev!=chainID1 && chainID2_prev==chainID2)
            {
                sequence[0]+='-';
                sequence[1]+=seqy[i2++];
                chainID2_prev=chainID2;
            }
            else
            {
                sequence[0]+=seqx[i1++];
                sequence[1]+=seqy[i2++];
                chainID1_prev=chainID1;
                chainID2_prev=chainID2;
            }
        }
        
    }
    map<string,string>().swap(chainID_map1);
    map<string,string>().swap(chainID_map2);
    chainID1.clear();
    chainID2.clear();
    chainID1_prev.clear();
    chainID2_prev.clear();
    return sequence[0].size();
}

/* extract pairwise sequence alignment from residue index vectors,
 * return length of alignment, including gap. */
int extract_aln_from_resi(vector<string> &sequence, char *seqx, char *seqy,
    const vector<string> resi_vec1, const vector<string> resi_vec2,
    const vector<int> xlen_vec, const vector<int> ylen_vec,
    const int chain_i, const int chain_j)
{
    sequence.clear();
    sequence.push_back("");
    sequence.push_back("");

    int i1=0; // positions in resi_vec1
    int i2=0; // positions in resi_vec2
    int xlen=xlen_vec[chain_i];
    int ylen=ylen_vec[chain_j];
    int i,j;
    for (i=0;i<chain_i;i++) i1+=xlen_vec[i];
    for (j=0;j<chain_j;j++) i2+=ylen_vec[j];

    i=j=0;
    while(i<xlen && j<ylen)
    {
        if (atoi(resi_vec1[i+i1].substr(0,4).c_str())<
            atoi(resi_vec2[j+i2].substr(0,4).c_str()))
        {
            sequence[0]+=seqx[i++];
            sequence[1]+='-';
        }
        else if (atoi(resi_vec1[i+i1].substr(0,4).c_str())>
                 atoi(resi_vec2[j+i2].substr(0,4).c_str()))
        {
            sequence[0]+='-';
            sequence[1]+=seqy[j++];
        }
        else
        {
            sequence[0]+=seqx[i++];
            sequence[1]+=seqy[j++];
        }
    }
    if (i<xlen && j==ylen)
    {
        for (i;i<xlen;i++)
        {
            sequence[0]+=seqx[i];
            sequence[1]+='-';
        }
    }
    else if (i==xlen && j<ylen)
    {
        for (j;j<ylen;j++)
        {
            sequence[0]+='-';
            sequence[1]+=seqy[j];
        }
    }
    return sequence[0].size();
}

/* param_set.h */
/* These functions implement d0 normalization. The d0 for final TM-score
 * output is implemented by parameter_set4final. For both RNA alignment
 * and protein alignment, using d0 set by parameter_set4search yields
 * slightly better results during initial alignment-superposition iteration.
 */

void parameter_set4search(const int xlen, const int ylen,
    double &D0_MIN, double &Lnorm,
    double &score_d8, double &d0, double &d0_search, double &dcu0)
{
    //parameter initialization for searching: D0_MIN, Lnorm, d0, d0_search, score_d8
    D0_MIN=0.5; 
    dcu0=4.25;                       //update 3.85-->4.25
 
    Lnorm=getmin(xlen, ylen);        //normalize TMscore by this in searching
    if (Lnorm<=19)                    //update 15-->19
        d0=0.168;                   //update 0.5-->0.168
    else d0=(1.24*pow((Lnorm*1.0-15), 1.0/3)-1.8);
    D0_MIN=d0+0.8;              //this should be moved to above
    d0=D0_MIN;                  //update: best for search    

    d0_search=d0;
    if (d0_search>8)   d0_search=8;
    if (d0_search<4.5) d0_search=4.5;

    score_d8=1.5*pow(Lnorm*1.0, 0.3)+3.5; //remove pairs with dis>d8 during search & final
}

void parameter_set4final_C3prime(const double len, double &D0_MIN,
    double &Lnorm, double &d0, double &d0_search)
{
    D0_MIN=0.3; 
 
    Lnorm=len;            //normalize TMscore by this in searching
    if(Lnorm<=11) d0=0.3;
    else if(Lnorm>11&&Lnorm<=15) d0=0.4;
    else if(Lnorm>15&&Lnorm<=19) d0=0.5;
    else if(Lnorm>19&&Lnorm<=23) d0=0.6;
    else if(Lnorm>23&&Lnorm<30)  d0=0.7;
    else d0=(0.6*pow((Lnorm*1.0-0.5), 1.0/2)-2.5);

    d0_search=d0;
    if (d0_search>8)   d0_search=8;
    if (d0_search<4.5) d0_search=4.5;
}

void parameter_set4final(const double len, double &D0_MIN, double &Lnorm,
    double &d0, double &d0_search, const int mol_type)
{
    if (mol_type>0) // RNA
    {
        parameter_set4final_C3prime(len, D0_MIN, Lnorm,
            d0, d0_search);
        return;
    }
    D0_MIN=0.5; 
 
    Lnorm=len;            //normalize TMscore by this in searching
    if (Lnorm<=21) d0=0.5;          
    else d0=(1.24*pow((Lnorm*1.0-15), 1.0/3)-1.8);
    if (d0<D0_MIN) d0=D0_MIN;   
    d0_search=d0;
    if (d0_search>8)   d0_search=8;
    if (d0_search<4.5) d0_search=4.5;
}

void parameter_set4scale(const int len, const double d_s, double &Lnorm,
    double &d0, double &d0_search)
{
    d0=d_s;          
    Lnorm=len;            //normalize TMscore by this in searching
    d0_search=d0;
    if (d0_search>8)   d0_search=8;
    if (d0_search<4.5) d0_search=4.5;  
}

/* NW.h */
/* Partial implementation of Needleman-Wunsch (NW) dynamic programming for
 * global alignment. The three NWDP_TM functions below are not complete
 * implementation of NW algorithm because gap jumping in the standard Gotoh
 * algorithm is not considered. Since the gap opening and gap extension is
 * the same, this is not a problem. This code was exploited in TM-align
 * because it is about 1.5 times faster than a complete NW implementation.
 * Nevertheless, if gap opening != gap extension shall be implemented in
 * the future, the Gotoh algorithm must be implemented. In rare scenarios,
 * it is also possible to have asymmetric alignment (i.e. 
 * TMalign A.pdb B.pdb and TMalign B.pdb A.pdb have different TM_A and TM_B
 * values) caused by the NWPD_TM implement.
 */

/* Input: score[1:len1, 1:len2], and gap_open
 * Output: j2i[1:len2] \in {1:len1} U {-1}
 * path[0:len1, 0:len2]=1,2,3, from diagonal, horizontal, vertical */
void NWDP_TM(double **score, bool **path, double **val,
    int len1, int len2, double gap_open, int j2i[])
{

    int i, j;
    double h, v, d;

    //initialization
    for(i=0; i<=len1; i++)
    {
        val[i][0]=0;
        //val[i][0]=i*gap_open;
        path[i][0]=false; //not from diagonal
    }

    for(j=0; j<=len2; j++)
    {
        val[0][j]=0;
        //val[0][j]=j*gap_open;
        path[0][j]=false; //not from diagonal
        j2i[j]=-1;    //all are not aligned, only use j2i[1:len2]
    }      


    //decide matrix and path
    for(i=1; i<=len1; i++)
    {
        for(j=1; j<=len2; j++)
        {
            d=val[i-1][j-1]+score[i][j]; //diagonal

            //symbol insertion in horizontal (= a gap in vertical)
            h=val[i-1][j];
            if(path[i-1][j]) h += gap_open; //aligned in last position

            //symbol insertion in vertical
            v=val[i][j-1];
            if(path[i][j-1]) v += gap_open; //aligned in last position


            if(d>=h && d>=v)
            {
                path[i][j]=true; //from diagonal
                val[i][j]=d;
            }
            else 
            {
                path[i][j]=false; //from horizontal
                if(v>=h) val[i][j]=v;
                else val[i][j]=h;
            }
        } //for i
    } //for j

    //trace back to extract the alignment
    i=len1;
    j=len2;
    while(i>0 && j>0)
    {
        if(path[i][j]) //from diagonal
        {
            j2i[j-1]=i-1;
            i--;
            j--;
        }
        else 
        {
            h=val[i-1][j];
            if(path[i-1][j]) h +=gap_open;

            v=val[i][j-1];
            if(path[i][j-1]) v +=gap_open;

            if(v>=h) j--;
            else i--;
        }
    }
}

/* Input: vectors x, y, rotation matrix t, u, scale factor d02, and gap_open
 * Output: j2i[1:len2] \in {1:len1} U {-1}
 * path[0:len1, 0:len2]=1,2,3, from diagonal, horizontal, vertical */
void NWDP_TM(bool **path, double **val, double **x, double **y,
    int len1, int len2, double t[3], double u[3][3],
    double d02, double gap_open, int j2i[])
{
    int i, j;
    double h, v, d;

    //initialization. use old val[i][0] and val[0][j] initialization
    //to minimize difference from TMalign fortran version
    for(i=0; i<=len1; i++)
    {
        val[i][0]=0;
        //val[i][0]=i*gap_open;
        path[i][0]=false; //not from diagonal
    }

    for(j=0; j<=len2; j++)
    {
        val[0][j]=0;
        //val[0][j]=j*gap_open;
        path[0][j]=false; //not from diagonal
        j2i[j]=-1;    //all are not aligned, only use j2i[1:len2]
    }      
    double xx[3], dij;


    //decide matrix and path
    for(i=1; i<=len1; i++)
    {
        transform(t, u, &x[i-1][0], xx);
        for(j=1; j<=len2; j++)
        {
            dij=dist(xx, &y[j-1][0]);    
            d=val[i-1][j-1] +  1.0/(1+dij/d02);

            //symbol insertion in horizontal (= a gap in vertical)
            h=val[i-1][j];
            if(path[i-1][j]) h += gap_open; //aligned in last position

            //symbol insertion in vertical
            v=val[i][j-1];
            if(path[i][j-1]) v += gap_open; //aligned in last position


            if(d>=h && d>=v)
            {
                path[i][j]=true; //from diagonal
                val[i][j]=d;
            }
            else 
            {
                path[i][j]=false; //from horizontal
                if(v>=h) val[i][j]=v;
                else val[i][j]=h;
            }
        } //for i
    } //for j

    //trace back to extract the alignment
    i=len1;
    j=len2;
    while(i>0 && j>0)
    {
        if(path[i][j]) //from diagonal
        {
            j2i[j-1]=i-1;
            i--;
            j--;
        }
        else 
        {
            h=val[i-1][j];
            if(path[i-1][j]) h +=gap_open;

            v=val[i][j-1];
            if(path[i][j-1]) v +=gap_open;

            if(v>=h) j--;
            else i--;
        }
    }
}

/* This is the same as the previous NWDP_TM, except for the lack of rotation
 * Input: vectors x, y, scale factor d02, and gap_open
 * Output: j2i[1:len2] \in {1:len1} U {-1}
 * path[0:len1, 0:len2]=1,2,3, from diagonal, horizontal, vertical */
void NWDP_SE(bool **path, double **val, double **x, double **y,
    int len1, int len2, double d02, double gap_open, int j2i[])
{
    int i, j;
    double h, v, d;

    for(i=0; i<=len1; i++)
    {
        val[i][0]=0;
        path[i][0]=false; //not from diagonal
    }

    for(j=0; j<=len2; j++)
    {
        val[0][j]=0;
        path[0][j]=false; //not from diagonal
        j2i[j]=-1;    //all are not aligned, only use j2i[1:len2]
    }      
    double dij;

    //decide matrix and path
    for(i=1; i<=len1; i++)
    {
        for(j=1; j<=len2; j++)
        {
            dij=dist(&x[i-1][0], &y[j-1][0]);    
            d=val[i-1][j-1] +  1.0/(1+dij/d02);

            //symbol insertion in horizontal (= a gap in vertical)
            h=val[i-1][j];
            if(path[i-1][j]) h += gap_open; //aligned in last position

            //symbol insertion in vertical
            v=val[i][j-1];
            if(path[i][j-1]) v += gap_open; //aligned in last position


            if(d>=h && d>=v)
            {
                path[i][j]=true; //from diagonal
                val[i][j]=d;
            }
            else 
            {
                path[i][j]=false; //from horizontal
                if(v>=h) val[i][j]=v;
                else val[i][j]=h;
            }
        } //for i
    } //for j

    //trace back to extract the alignment
    i=len1;
    j=len2;
    while(i>0 && j>0)
    {
        if(path[i][j]) //from diagonal
        {
            j2i[j-1]=i-1;
            i--;
            j--;
        }
        else 
        {
            h=val[i-1][j];
            if(path[i-1][j]) h +=gap_open;

            v=val[i][j-1];
            if(path[i][j-1]) v +=gap_open;

            if(v>=h) j--;
            else i--;
        }
    }
}

/* +ss
 * Input: secondary structure secx, secy, and gap_open
 * Output: j2i[1:len2] \in {1:len1} U {-1}
 * path[0:len1, 0:len2]=1,2,3, from diagonal, horizontal, vertical */
void NWDP_TM(bool **path, double **val, const char *secx, const char *secy,
    const int len1, const int len2, const double gap_open, int j2i[])
{

    int i, j;
    double h, v, d;

    //initialization
    for(i=0; i<=len1; i++)
    {
        val[i][0]=0;
        //val[i][0]=i*gap_open;
        path[i][0]=false; //not from diagonal
    }

    for(j=0; j<=len2; j++)
    {
        val[0][j]=0;
        //val[0][j]=j*gap_open;
        path[0][j]=false; //not from diagonal
        j2i[j]=-1;    //all are not aligned, only use j2i[1:len2]
    }      

    //decide matrix and path
    for(i=1; i<=len1; i++)
    {
        for(j=1; j<=len2; j++)
        {
            d=val[i-1][j-1] + 1.0*(secx[i-1]==secy[j-1]);

            //symbol insertion in horizontal (= a gap in vertical)
            h=val[i-1][j];
            if(path[i-1][j]) h += gap_open; //aligned in last position

            //symbol insertion in vertical
            v=val[i][j-1];
            if(path[i][j-1]) v += gap_open; //aligned in last position

            if(d>=h && d>=v)
            {
                path[i][j]=true; //from diagonal
                val[i][j]=d;
            }
            else 
            {
                path[i][j]=false; //from horizontal
                if(v>=h) val[i][j]=v;
                else val[i][j]=h;
            }
        } //for i
    } //for j

    //trace back to extract the alignment
    i=len1;
    j=len2;
    while(i>0 && j>0)
    {
        if(path[i][j]) //from diagonal
        {
            j2i[j-1]=i-1;
            i--;
            j--;
        }
        else 
        {
            h=val[i-1][j];
            if(path[i-1][j]) h +=gap_open;

            v=val[i][j-1];
            if(path[i][j-1]) v +=gap_open;

            if(v>=h) j--;
            else i--;
        }
    }
}

/* Kabsch.h */
/**************************************************************************
Implemetation of Kabsch algoritm for finding the best rotation matrix
---------------------------------------------------------------------------
x    - x(i,m) are coordinates of atom m in set x            (input)
y    - y(i,m) are coordinates of atom m in set y            (input)
n    - n is number of atom pairs                            (input)
mode  - 0:calculate rms only                                (input)
1:calculate u,t only                                (takes medium)
2:calculate rms,u,t                                 (takes longer)
rms   - sum of w*(ux+t-y)**2 over all atom pairs            (output)
u    - u(i,j) is   rotation  matrix for best superposition  (output)
t    - t(i)   is translation vector for best superposition  (output)
**************************************************************************/
bool Kabsch(double **x, double **y, int n, int mode, double *rms,
    double t[3], double u[3][3])
{
    int i, j, m, m1, l, k;
    double e0, rms1, d, h, g;
    double cth, sth, sqrth, p, det, sigma;
    double xc[3], yc[3];
    double a[3][3], b[3][3], r[3][3], e[3], rr[6], ss[6];
    double sqrt3 = 1.73205080756888, tol = 0.01;
    int ip[] = { 0, 1, 3, 1, 2, 4, 3, 4, 5 };
    int ip2312[] = { 1, 2, 0, 1 };

    int a_failed = 0, b_failed = 0;
    double epsilon = 0.00000001;

    //initialization
    *rms = 0;
    rms1 = 0;
    e0 = 0;
    double c1[3], c2[3];
    double s1[3], s2[3];
    double sx[3], sy[3], sz[3];
    for (i = 0; i < 3; i++)
    {
        s1[i] = 0.0;
        s2[i] = 0.0;

        sx[i] = 0.0;
        sy[i] = 0.0;
        sz[i] = 0.0;
    }

    for (i = 0; i<3; i++)
    {
        xc[i] = 0.0;
        yc[i] = 0.0;
        t[i] = 0.0;
        for (j = 0; j<3; j++)
        {
            u[i][j] = 0.0;
            r[i][j] = 0.0;
            a[i][j] = 0.0;
            if (i == j)
            {
                u[i][j] = 1.0;
                a[i][j] = 1.0;
            }
        }
    }

    if (n<1) return false;

    //compute centers for vector sets x, y
    for (i = 0; i<n; i++)
    {
        for (j = 0; j < 3; j++)
        {
            c1[j] = x[i][j];
            c2[j] = y[i][j];

            s1[j] += c1[j];
            s2[j] += c2[j];
        }

        for (j = 0; j < 3; j++)
        {
            sx[j] += c1[0] * c2[j];
            sy[j] += c1[1] * c2[j];
            sz[j] += c1[2] * c2[j];
        }
    }
    for (i = 0; i < 3; i++)
    {
        xc[i] = s1[i] / n;
        yc[i] = s2[i] / n;
    }
    if (mode == 2 || mode == 0)
        for (int mm = 0; mm < n; mm++)
            for (int nn = 0; nn < 3; nn++)
                e0 += (x[mm][nn] - xc[nn]) * (x[mm][nn] - xc[nn]) + 
                      (y[mm][nn] - yc[nn]) * (y[mm][nn] - yc[nn]);
    for (j = 0; j < 3; j++)
    {
        r[j][0] = sx[j] - s1[0] * s2[j] / n;
        r[j][1] = sy[j] - s1[1] * s2[j] / n;
        r[j][2] = sz[j] - s1[2] * s2[j] / n;
    }

    //compute determinant of matrix r
    det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])\
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])\
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    sigma = det;

    //compute tras(r)*r
    m = 0;
    for (j = 0; j<3; j++)
    {
        for (i = 0; i <= j; i++)
        {
            rr[m] = r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j];
            m++;
        }
    }

    double spur = (rr[0] + rr[2] + rr[5]) / 3.0;
    double cof = (((((rr[2] * rr[5] - rr[4] * rr[4]) + rr[0] * rr[5])\
        - rr[3] * rr[3]) + rr[0] * rr[2]) - rr[1] * rr[1]) / 3.0;
    det = det*det;

    for (i = 0; i<3; i++) e[i] = spur;

    if (spur>0)
    {
        d = spur*spur;
        h = d - cof;
        g = (spur*cof - det) / 2.0 - spur*h;

        if (h>0)
        {
            sqrth = sqrt(h);
            d = h*h*h - g*g;
            if (d<0.0) d = 0.0;
            d = atan2(sqrt(d), -g) / 3.0;
            cth = sqrth * cos(d);
            sth = sqrth*sqrt3*sin(d);
            e[0] = (spur + cth) + cth;
            e[1] = (spur - cth) + sth;
            e[2] = (spur - cth) - sth;

            if (mode != 0)
            {//compute a                
                for (l = 0; l<3; l = l + 2)
                {
                    d = e[l];
                    ss[0] = (d - rr[2]) * (d - rr[5]) - rr[4] * rr[4];
                    ss[1] = (d - rr[5]) * rr[1] + rr[3] * rr[4];
                    ss[2] = (d - rr[0]) * (d - rr[5]) - rr[3] * rr[3];
                    ss[3] = (d - rr[2]) * rr[3] + rr[1] * rr[4];
                    ss[4] = (d - rr[0]) * rr[4] + rr[1] * rr[3];
                    ss[5] = (d - rr[0]) * (d - rr[2]) - rr[1] * rr[1];

                    if (fabs(ss[0]) <= epsilon) ss[0] = 0.0;
                    if (fabs(ss[1]) <= epsilon) ss[1] = 0.0;
                    if (fabs(ss[2]) <= epsilon) ss[2] = 0.0;
                    if (fabs(ss[3]) <= epsilon) ss[3] = 0.0;
                    if (fabs(ss[4]) <= epsilon) ss[4] = 0.0;
                    if (fabs(ss[5]) <= epsilon) ss[5] = 0.0;

                    if (fabs(ss[0]) >= fabs(ss[2]))
                    {
                        j = 0;
                        if (fabs(ss[0]) < fabs(ss[5])) j = 2;
                    }
                    else if (fabs(ss[2]) >= fabs(ss[5])) j = 1;
                    else j = 2;

                    d = 0.0;
                    j = 3 * j;
                    for (i = 0; i<3; i++)
                    {
                        k = ip[i + j];
                        a[i][l] = ss[k];
                        d = d + ss[k] * ss[k];
                    }


                    //if( d > 0.0 ) d = 1.0 / sqrt(d);
                    if (d > epsilon) d = 1.0 / sqrt(d);
                    else d = 0.0;
                    for (i = 0; i<3; i++) a[i][l] = a[i][l] * d;
                }//for l

                d = a[0][0] * a[0][2] + a[1][0] * a[1][2] + a[2][0] * a[2][2];
                if ((e[0] - e[1]) >(e[1] - e[2]))
                {
                    m1 = 2;
                    m = 0;
                }
                else
                {
                    m1 = 0;
                    m = 2;
                }
                p = 0;
                for (i = 0; i<3; i++)
                {
                    a[i][m1] = a[i][m1] - d*a[i][m];
                    p = p + a[i][m1] * a[i][m1];
                }
                if (p <= tol)
                {
                    p = 1.0;
                    for (i = 0; i<3; i++)
                    {
                        if (p < fabs(a[i][m])) continue;
                        p = fabs(a[i][m]);
                        j = i;
                    }
                    k = ip2312[j];
                    l = ip2312[j + 1];
                    p = sqrt(a[k][m] * a[k][m] + a[l][m] * a[l][m]);
                    if (p > tol)
                    {
                        a[j][m1] = 0.0;
                        a[k][m1] = -a[l][m] / p;
                        a[l][m1] = a[k][m] / p;
                    }
                    else a_failed = 1;
                }//if p<=tol
                else
                {
                    p = 1.0 / sqrt(p);
                    for (i = 0; i<3; i++) a[i][m1] = a[i][m1] * p;
                }//else p<=tol  
                if (a_failed != 1)
                {
                    a[0][1] = a[1][2] * a[2][0] - a[1][0] * a[2][2];
                    a[1][1] = a[2][2] * a[0][0] - a[2][0] * a[0][2];
                    a[2][1] = a[0][2] * a[1][0] - a[0][0] * a[1][2];
                }
            }//if(mode!=0)       
        }//h>0

        //compute b anyway
        if (mode != 0 && a_failed != 1)//a is computed correctly
        {
            //compute b
            for (l = 0; l<2; l++)
            {
                d = 0.0;
                for (i = 0; i<3; i++)
                {
                    b[i][l] = r[i][0] * a[0][l] + 
                              r[i][1] * a[1][l] + r[i][2] * a[2][l];
                    d = d + b[i][l] * b[i][l];
                }
                //if( d > 0 ) d = 1.0 / sqrt(d);
                if (d > epsilon) d = 1.0 / sqrt(d);
                else d = 0.0;
                for (i = 0; i<3; i++) b[i][l] = b[i][l] * d;
            }
            d = b[0][0] * b[0][1] + b[1][0] * b[1][1] + b[2][0] * b[2][1];
            p = 0.0;

            for (i = 0; i<3; i++)
            {
                b[i][1] = b[i][1] - d*b[i][0];
                p += b[i][1] * b[i][1];
            }

            if (p <= tol)
            {
                p = 1.0;
                for (i = 0; i<3; i++)
                {
                    if (p<fabs(b[i][0])) continue;
                    p = fabs(b[i][0]);
                    j = i;
                }
                k = ip2312[j];
                l = ip2312[j + 1];
                p = sqrt(b[k][0] * b[k][0] + b[l][0] * b[l][0]);
                if (p > tol)
                {
                    b[j][1] = 0.0;
                    b[k][1] = -b[l][0] / p;
                    b[l][1] = b[k][0] / p;
                }
                else b_failed = 1;
            }//if( p <= tol )
            else
            {
                p = 1.0 / sqrt(p);
                for (i = 0; i<3; i++) b[i][1] = b[i][1] * p;
            }
            if (b_failed != 1)
            {
                b[0][2] = b[1][0] * b[2][1] - b[1][1] * b[2][0];
                b[1][2] = b[2][0] * b[0][1] - b[2][1] * b[0][0];
                b[2][2] = b[0][0] * b[1][1] - b[0][1] * b[1][0];
                //compute u
                for (i = 0; i<3; i++)
                    for (j = 0; j<3; j++)
                        u[i][j] = b[i][0] * a[j][0] + 
                                  b[i][1] * a[j][1] + b[i][2] * a[j][2];
            }

            //compute t
            for (i = 0; i<3; i++)
                t[i] = ((yc[i] - u[i][0] * xc[0]) - u[i][1] * xc[1]) - 
                                                    u[i][2] * xc[2];
        }//if(mode!=0 && a_failed!=1)
    }//spur>0
    else //just compute t and errors
    {
        //compute t
        for (i = 0; i<3; i++)
            t[i] = ((yc[i] - u[i][0] * xc[0]) - u[i][1] * xc[1]) - 
                                                u[i][2] * xc[2];
    }//else spur>0 

    //compute rms
    for (i = 0; i<3; i++)
    {
        if (e[i] < 0) e[i] = 0;
        e[i] = sqrt(e[i]);
    }
    d = e[2];
    if (sigma < 0.0) d = -d;
    d = (d + e[1]) + e[0];

    if (mode == 2 || mode == 0)
    {
        rms1 = (e0 - d) - d;
        if (rms1 < 0.0) rms1 = 0.0;
    }

    *rms = rms1;
    return true;
}

/* TMalign.h" */
/* Functions for the core TMalign algorithm, including the entry function
 * TMalign_main */

//     1, collect those residues with dis<d;
//     2, calculate TMscore
int score_fun8( double **xa, double **ya, int n_ali, double d, int i_ali[],
    double *score1, int score_sum_method, const double Lnorm, 
    const double score_d8, const double d0)
{
    double score_sum=0, di;
    double d_tmp=d*d;
    double d02=d0*d0;
    double score_d8_cut = score_d8*score_d8;
    
    int i, n_cut, inc=0;

    while(1)
    {
        n_cut=0;
        score_sum=0;
        for(i=0; i<n_ali; i++)
        {
            di = dist(xa[i], ya[i]);
            if(di<d_tmp)
            {
                i_ali[n_cut]=i;
                n_cut++;
            }
            if(score_sum_method==8)
            {                
                if(di<=score_d8_cut) score_sum += 1/(1+di/d02);
            }
            else score_sum += 1/(1+di/d02);
        }
        //there are not enough feasible pairs, relieve the threshold         
        if(n_cut<3 && n_ali>3)
        {
            inc++;
            double dinc=(d+inc*0.5);
            d_tmp = dinc * dinc;
        }
        else break;
    }  

    *score1=score_sum/Lnorm;
    return n_cut;
}

int score_fun8_standard(double **xa, double **ya, int n_ali, double d,
    int i_ali[], double *score1, int score_sum_method,
    double score_d8, double d0)
{
    double score_sum = 0, di;
    double d_tmp = d*d;
    double d02 = d0*d0;
    double score_d8_cut = score_d8*score_d8;

    int i, n_cut, inc = 0;
    while (1)
    {
        n_cut = 0;
        score_sum = 0;
        for (i = 0; i<n_ali; i++)
        {
            di = dist(xa[i], ya[i]);
            if (di<d_tmp)
            {
                i_ali[n_cut] = i;
                n_cut++;
            }
            if (score_sum_method == 8)
            {
                if (di <= score_d8_cut) score_sum += 1 / (1 + di / d02);
            }
            else
            {
                score_sum += 1 / (1 + di / d02);
            }
        }
        //there are not enough feasible pairs, relieve the threshold         
        if (n_cut<3 && n_ali>3)
        {
            inc++;
            double dinc = (d + inc*0.5);
            d_tmp = dinc * dinc;
        }
        else break;
    }

    *score1 = score_sum / n_ali;
    return n_cut;
}

double TMscore8_search(double **r1, double **r2, double **xtm, double **ytm,
    double **xt, int Lali, double t0[3], double u0[3][3], int simplify_step,
    int score_sum_method, double *Rcomm, double local_d0_search, double Lnorm,
    double score_d8, double d0)
{
    int i, m;
    double score_max, score, rmsd;    
    const int kmax=Lali;    
    int k_ali[kmax], ka, k;
    double t[3];
    double u[3][3];
    double d;
    

    //iterative parameters
    int n_it=20;            //maximum number of iterations
    int n_init_max=6; //maximum number of different fragment length 
    int L_ini[n_init_max];  //fragment lengths, Lali, Lali/2, Lali/4 ... 4   
    int L_ini_min=4;
    if(Lali<L_ini_min) L_ini_min=Lali;   

    int n_init=0, i_init;      
    for(i=0; i<n_init_max-1; i++)
    {
        n_init++;
        L_ini[i]=(int) (Lali/pow(2.0, (double) i));
        if(L_ini[i]<=L_ini_min)
        {
            L_ini[i]=L_ini_min;
            break;
        }
    }
    if(i==n_init_max-1)
    {
        n_init++;
        L_ini[i]=L_ini_min;
    }
    
    score_max=-1;
    //find the maximum score starting from local structures superposition
    int i_ali[kmax], n_cut;
    int L_frag; //fragment length
    int iL_max; //maximum starting position for the fragment
    
    for(i_init=0; i_init<n_init; i_init++)
    {
        L_frag=L_ini[i_init];
        iL_max=Lali-L_frag;
      
        i=0;   
        while(1)
        {
            //extract the fragment starting from position i 
            ka=0;
            for(k=0; k<L_frag; k++)
            {
                int kk=k+i;
                r1[k][0]=xtm[kk][0];  
                r1[k][1]=xtm[kk][1]; 
                r1[k][2]=xtm[kk][2];   
                
                r2[k][0]=ytm[kk][0];  
                r2[k][1]=ytm[kk][1]; 
                r2[k][2]=ytm[kk][2];
                
                k_ali[ka]=kk;
                ka++;
            }
            
            //extract rotation matrix based on the fragment
            Kabsch(r1, r2, L_frag, 1, &rmsd, t, u);
            if (simplify_step != 1)
                *Rcomm = 0;
            do_rotation(xtm, xt, Lali, t, u);
            
            //get subsegment of this fragment
            d = local_d0_search - 1;
            n_cut=score_fun8(xt, ytm, Lali, d, i_ali, &score, 
                score_sum_method, Lnorm, score_d8, d0);
            if(score>score_max)
            {
                score_max=score;
                
                //save the rotation matrix
                for(k=0; k<3; k++)
                {
                    t0[k]=t[k];
                    u0[k][0]=u[k][0];
                    u0[k][1]=u[k][1];
                    u0[k][2]=u[k][2];
                }
            }
            
            //try to extend the alignment iteratively            
            d = local_d0_search + 1;
            for(int it=0; it<n_it; it++)            
            {
                ka=0;
                for(k=0; k<n_cut; k++)
                {
                    m=i_ali[k];
                    r1[k][0]=xtm[m][0];  
                    r1[k][1]=xtm[m][1]; 
                    r1[k][2]=xtm[m][2];
                    
                    r2[k][0]=ytm[m][0];  
                    r2[k][1]=ytm[m][1]; 
                    r2[k][2]=ytm[m][2];
                    
                    k_ali[ka]=m;
                    ka++;
                } 
                //extract rotation matrix based on the fragment                
                Kabsch(r1, r2, n_cut, 1, &rmsd, t, u);
                do_rotation(xtm, xt, Lali, t, u);
                n_cut=score_fun8(xt, ytm, Lali, d, i_ali, &score, 
                    score_sum_method, Lnorm, score_d8, d0);
                if(score>score_max)
                {
                    score_max=score;

                    //save the rotation matrix
                    for(k=0; k<3; k++)
                    {
                        t0[k]=t[k];
                        u0[k][0]=u[k][0];
                        u0[k][1]=u[k][1];
                        u0[k][2]=u[k][2];
                    }                     
                }
                
                //check if it converges            
                if(n_cut==ka)
                {                
                    for(k=0; k<n_cut; k++)
                    {
                        if(i_ali[k]!=k_ali[k]) break;
                    }
                    if(k==n_cut) break;
                }                                                               
            } //for iteration            

            if(i<iL_max)
            {
                i=i+simplify_step; //shift the fragment        
                if(i>iL_max) i=iL_max;  //do this to use the last missed fragment
            }
            else if(i>=iL_max) break;
        }//while(1)
        //end of one fragment
    }//for(i_init
    return score_max;
}


double TMscore8_search_standard( double **r1, double **r2,
    double **xtm, double **ytm, double **xt, int Lali,
    double t0[3], double u0[3][3], int simplify_step, int score_sum_method,
    double *Rcomm, double local_d0_search, double score_d8, double d0)
{
    int i, m;
    double score_max, score, rmsd;
    const int kmax = Lali;
    int k_ali[kmax], ka, k;
    double t[3];
    double u[3][3];
    double d;

    //iterative parameters
    int n_it = 20;            //maximum number of iterations
    int n_init_max = 6; //maximum number of different fragment length 
    int L_ini[n_init_max];  //fragment lengths, Lali, Lali/2, Lali/4 ... 4   
    int L_ini_min = 4;
    if (Lali<L_ini_min) L_ini_min = Lali;

    int n_init = 0, i_init;
    for (i = 0; i<n_init_max - 1; i++)
    {
        n_init++;
        L_ini[i] = (int)(Lali / pow(2.0, (double)i));
        if (L_ini[i] <= L_ini_min)
        {
            L_ini[i] = L_ini_min;
            break;
        }
    }
    if (i == n_init_max - 1)
    {
        n_init++;
        L_ini[i] = L_ini_min;
    }

    score_max = -1;
    //find the maximum score starting from local structures superposition
    int i_ali[kmax], n_cut;
    int L_frag; //fragment length
    int iL_max; //maximum starting position for the fragment

    for (i_init = 0; i_init<n_init; i_init++)
    {
        L_frag = L_ini[i_init];
        iL_max = Lali - L_frag;

        i = 0;
        while (1)
        {
            //extract the fragment starting from position i 
            ka = 0;
            for (k = 0; k<L_frag; k++)
            {
                int kk = k + i;
                r1[k][0] = xtm[kk][0];
                r1[k][1] = xtm[kk][1];
                r1[k][2] = xtm[kk][2];

                r2[k][0] = ytm[kk][0];
                r2[k][1] = ytm[kk][1];
                r2[k][2] = ytm[kk][2];

                k_ali[ka] = kk;
                ka++;
            }
            //extract rotation matrix based on the fragment
            Kabsch(r1, r2, L_frag, 1, &rmsd, t, u);
            if (simplify_step != 1)
                *Rcomm = 0;
            do_rotation(xtm, xt, Lali, t, u);

            //get subsegment of this fragment
            d = local_d0_search - 1;
            n_cut = score_fun8_standard(xt, ytm, Lali, d, i_ali, &score,
                score_sum_method, score_d8, d0);

            if (score>score_max)
            {
                score_max = score;

                //save the rotation matrix
                for (k = 0; k<3; k++)
                {
                    t0[k] = t[k];
                    u0[k][0] = u[k][0];
                    u0[k][1] = u[k][1];
                    u0[k][2] = u[k][2];
                }
            }

            //try to extend the alignment iteratively            
            d = local_d0_search + 1;
            for (int it = 0; it<n_it; it++)
            {
                ka = 0;
                for (k = 0; k<n_cut; k++)
                {
                    m = i_ali[k];
                    r1[k][0] = xtm[m][0];
                    r1[k][1] = xtm[m][1];
                    r1[k][2] = xtm[m][2];

                    r2[k][0] = ytm[m][0];
                    r2[k][1] = ytm[m][1];
                    r2[k][2] = ytm[m][2];

                    k_ali[ka] = m;
                    ka++;
                }
                //extract rotation matrix based on the fragment                
                Kabsch(r1, r2, n_cut, 1, &rmsd, t, u);
                do_rotation(xtm, xt, Lali, t, u);
                n_cut = score_fun8_standard(xt, ytm, Lali, d, i_ali, &score,
                    score_sum_method, score_d8, d0);
                if (score>score_max)
                {
                    score_max = score;

                    //save the rotation matrix
                    for (k = 0; k<3; k++)
                    {
                        t0[k] = t[k];
                        u0[k][0] = u[k][0];
                        u0[k][1] = u[k][1];
                        u0[k][2] = u[k][2];
                    }
                }

                //check if it converges            
                if (n_cut == ka)
                {
                    for (k = 0; k<n_cut; k++)
                    {
                        if (i_ali[k] != k_ali[k]) break;
                    }
                    if (k == n_cut) break;
                }
            } //for iteration            

            if (i<iL_max)
            {
                i = i + simplify_step; //shift the fragment        
                if (i>iL_max) i = iL_max;  //do this to use the last missed fragment
            }
            else if (i >= iL_max) break;
        }//while(1)
        //end of one fragment
    }//for(i_init
    return score_max;
}

//Comprehensive TMscore search engine
// input:   two vector sets: x, y
//          an alignment invmap0[] between x and y
//          simplify_step: 1 or 40 or other integers
//          score_sum_method: 0 for score over all pairs
//                            8 for socre over the pairs with dist<score_d8
// output:  the best rotaion matrix t, u that results in highest TMscore
double detailed_search(double **r1, double **r2, double **xtm, double **ytm,
    double **xt, double **x, double **y, int xlen, int ylen, 
    int invmap0[], double t[3], double u[3][3], int simplify_step,
    int score_sum_method, double local_d0_search, double Lnorm,
    double score_d8, double d0)
{
    //x is model, y is template, try to superpose onto y
    int i, j, k;     
    double tmscore;
    double rmsd;

    k=0;
    for(i=0; i<ylen; i++) 
    {
        j=invmap0[i];
        if(j>=0) //aligned
        {
            xtm[k][0]=x[j][0];
            xtm[k][1]=x[j][1];
            xtm[k][2]=x[j][2];
                
            ytm[k][0]=y[i][0];
            ytm[k][1]=y[i][1];
            ytm[k][2]=y[i][2];
            k++;
        }
    }

    //detailed search 40-->1
    tmscore = TMscore8_search(r1, r2, xtm, ytm, xt, k, t, u, simplify_step,
        score_sum_method, &rmsd, local_d0_search, Lnorm, score_d8, d0);
    return tmscore;
}

double detailed_search_standard( double **r1, double **r2,
    double **xtm, double **ytm, double **xt, double **x, double **y,
    int xlen, int ylen, int invmap0[], double t[3], double u[3][3],
    int simplify_step, int score_sum_method, double local_d0_search,
    const bool& bNormalize, double Lnorm, double score_d8, double d0)
{
    //x is model, y is template, try to superpose onto y
    int i, j, k;     
    double tmscore;
    double rmsd;

    k=0;
    for(i=0; i<ylen; i++) 
    {
        j=invmap0[i];
        if(j>=0) //aligned
        {
            xtm[k][0]=x[j][0];
            xtm[k][1]=x[j][1];
            xtm[k][2]=x[j][2];
                
            ytm[k][0]=y[i][0];
            ytm[k][1]=y[i][1];
            ytm[k][2]=y[i][2];
            k++;
        }
    }

    //detailed search 40-->1
    tmscore = TMscore8_search_standard( r1, r2, xtm, ytm, xt, k, t, u,
        simplify_step, score_sum_method, &rmsd, local_d0_search, score_d8, d0);
    if (bNormalize)// "-i", to use standard_TMscore, then bNormalize=true, else bNormalize=false; 
        tmscore = tmscore * k / Lnorm;

    return tmscore;
}

//compute the score quickly in three iterations
double get_score_fast( double **r1, double **r2, double **xtm, double **ytm,
    double **x, double **y, int xlen, int ylen, int invmap[],
    double d0, double d0_search, double t[3], double u[3][3])
{
    double rms, tmscore, tmscore1, tmscore2;
    int i, j, k;

    k=0;
    for(j=0; j<ylen; j++)
    {
        i=invmap[j];
        if(i>=0)
        {
            r1[k][0]=x[i][0];
            r1[k][1]=x[i][1];
            r1[k][2]=x[i][2];

            r2[k][0]=y[j][0];
            r2[k][1]=y[j][1];
            r2[k][2]=y[j][2];
            
            xtm[k][0]=x[i][0];
            xtm[k][1]=x[i][1];
            xtm[k][2]=x[i][2];
            
            ytm[k][0]=y[j][0];
            ytm[k][1]=y[j][1];
            ytm[k][2]=y[j][2];                  
            
            k++;
        }
        else if(i!=-1) PrintErrorAndQuit("Wrong map!\n");
    }
    Kabsch(r1, r2, k, 1, &rms, t, u);
    
    //evaluate score   
    double di;
    const int len=k;
    double dis[len];    
    double d00=d0_search;
    double d002=d00*d00;
    double d02=d0*d0;
    
    int n_ali=k;
    double xrot[3];
    tmscore=0;
    for(k=0; k<n_ali; k++)
    {
        transform(t, u, &xtm[k][0], xrot);        
        di=dist(xrot, &ytm[k][0]);
        dis[k]=di;
        tmscore += 1/(1+di/d02);
    }
    
   
   
   //second iteration 
    double d002t=d002;
    while(1)
    {
        j=0;
        for(k=0; k<n_ali; k++)
        {            
            if(dis[k]<=d002t)
            {
                r1[j][0]=xtm[k][0];
                r1[j][1]=xtm[k][1];
                r1[j][2]=xtm[k][2];
                
                r2[j][0]=ytm[k][0];
                r2[j][1]=ytm[k][1];
                r2[j][2]=ytm[k][2];
                
                j++;
            }
        }
        //there are not enough feasible pairs, relieve the threshold 
        if(j<3 && n_ali>3) d002t += 0.5;
        else break;
    }
    
    if(n_ali!=j)
    {
        Kabsch(r1, r2, j, 1, &rms, t, u);
        tmscore1=0;
        for(k=0; k<n_ali; k++)
        {
            transform(t, u, &xtm[k][0], xrot);        
            di=dist(xrot, &ytm[k][0]);
            dis[k]=di;
            tmscore1 += 1/(1+di/d02);
        }
        
        //third iteration
        d002t=d002+1;
       
        while(1)
        {
            j=0;
            for(k=0; k<n_ali; k++)
            {            
                if(dis[k]<=d002t)
                {
                    r1[j][0]=xtm[k][0];
                    r1[j][1]=xtm[k][1];
                    r1[j][2]=xtm[k][2];
                    
                    r2[j][0]=ytm[k][0];
                    r2[j][1]=ytm[k][1];
                    r2[j][2]=ytm[k][2];
                                        
                    j++;
                }
            }
            //there are not enough feasible pairs, relieve the threshold 
            if(j<3 && n_ali>3) d002t += 0.5;
            else break;
        }

        //evaluate the score
        Kabsch(r1, r2, j, 1, &rms, t, u);
        tmscore2=0;
        for(k=0; k<n_ali; k++)
        {
            transform(t, u, &xtm[k][0], xrot);
            di=dist(xrot, &ytm[k][0]);
            tmscore2 += 1/(1+di/d02);
        }    
    }
    else
    {
        tmscore1=tmscore;
        tmscore2=tmscore;
    }
    
    if(tmscore1>=tmscore) tmscore=tmscore1;
    if(tmscore2>=tmscore) tmscore=tmscore2;
    return tmscore; // no need to normalize this score because it will not be used for latter scoring
}


//perform gapless threading to find the best initial alignment
//input: x, y, xlen, ylen
//output: y2x0 stores the best alignment: e.g., 
//y2x0[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
double get_initial(double **r1, double **r2, double **xtm, double **ytm,
    double **x, double **y, int xlen, int ylen, int *y2x,
    double d0, double d0_search, const bool fast_opt,
    double t[3], double u[3][3])
{
    int min_len=getmin(xlen, ylen);
    if(min_len<3) PrintErrorAndQuit("Sequence is too short <3!\n");
    
    int min_ali= min_len/2;              //minimum size of considered fragment 
    if(min_ali<=5)  min_ali=5;    
    int n1, n2;
    n1 = -ylen+min_ali; 
    n2 = xlen-min_ali;

    int i, j, k, k_best;
    double tmscore, tmscore_max=-1;

    k_best=n1;
    for(k=n1; k<=n2; k+=(fast_opt)?5:1)
    {
        //get the map
        for(j=0; j<ylen; j++)
        {
            i=j+k;
            if(i>=0 && i<xlen) y2x[j]=i;
            else y2x[j]=-1;
        }
        
        //evaluate the map quickly in three iterations
        //this is not real tmscore, it is used to evaluate the goodness of the initial alignment
        tmscore=get_score_fast(r1, r2, xtm, ytm,
            x, y, xlen, ylen, y2x, d0,d0_search, t, u);
        if(tmscore>=tmscore_max)
        {
            tmscore_max=tmscore;
            k_best=k;
        }
    }
    
    //extract the best map
    k=k_best;
    for(j=0; j<ylen; j++)
    {
        i=j+k;
        if(i>=0 && i<xlen) y2x[j]=i;
        else y2x[j]=-1;
    }    

    return tmscore_max;
}

void smooth(int *sec, int len)
{
    int i, j;
    //smooth single  --x-- => -----
    for (i=2; i<len-2; i++)
    {
        if(sec[i]==2 || sec[i]==4)
        {
            j=sec[i];
            if (sec[i-2]!=j && sec[i-1]!=j && sec[i+1]!=j && sec[i+2]!=j)
                sec[i]=1;
        }
    }

    //   smooth double 
    //   --xx-- => ------
    for (i=0; i<len-5; i++)
    {
        //helix
        if (sec[i]!=2   && sec[i+1]!=2 && sec[i+2]==2 && sec[i+3]==2 &&
            sec[i+4]!=2 && sec[i+5]!= 2)
        {
            sec[i+2]=1;
            sec[i+3]=1;
        }

        //beta
        if (sec[i]!=4   && sec[i+1]!=4 && sec[i+2]==4 && sec[i+3]==4 &&
            sec[i+4]!=4 && sec[i+5]!= 4)
        {
            sec[i+2]=1;
            sec[i+3]=1;
        }
    }

    //smooth connect
    for (i=0; i<len-2; i++)
    {        
        if (sec[i]==2 && sec[i+1]!=2 && sec[i+2]==2) sec[i+1]=2;
        else if(sec[i]==4 && sec[i+1]!=4 && sec[i+2]==4) sec[i+1]=4;
    }

}

char sec_str(double dis13, double dis14, double dis15,
            double dis24, double dis25, double dis35)
{
    char s='C';
    
    double delta=2.1;
    if (fabs(dis15-6.37)<delta && fabs(dis14-5.18)<delta && 
        fabs(dis25-5.18)<delta && fabs(dis13-5.45)<delta &&
        fabs(dis24-5.45)<delta && fabs(dis35-5.45)<delta)
    {
        s='H'; //helix                        
        return s;
    }

    delta=1.42;
    if (fabs(dis15-13  )<delta && fabs(dis14-10.4)<delta &&
        fabs(dis25-10.4)<delta && fabs(dis13-6.1 )<delta &&
        fabs(dis24-6.1 )<delta && fabs(dis35-6.1 )<delta)
    {
        s='E'; //strand
        return s;
    }

    if (dis15 < 8) s='T'; //turn
    return s;
}


/* secondary structure assignment for protein:
 * 1->coil, 2->helix, 3->turn, 4->strand */
void make_sec(double **x, int len, char *sec)
{
    int j1, j2, j3, j4, j5;
    double d13, d14, d15, d24, d25, d35;
    for(int i=0; i<len; i++)
    {     
        sec[i]='C';
        j1=i-2;
        j2=i-1;
        j3=i;
        j4=i+1;
        j5=i+2;        
        
        if(j1>=0 && j5<len)
        {
            d13=sqrt(dist(x[j1], x[j3]));
            d14=sqrt(dist(x[j1], x[j4]));
            d15=sqrt(dist(x[j1], x[j5]));
            d24=sqrt(dist(x[j2], x[j4]));
            d25=sqrt(dist(x[j2], x[j5]));
            d35=sqrt(dist(x[j3], x[j5]));
            sec[i]=sec_str(d13, d14, d15, d24, d25, d35);            
        }    
    } 
    sec[len]=0;
}

/* a c d b: a paired to b, c paired to d */
bool overlap(const int a1,const int b1,const int c1,const int d1,
             const int a2,const int b2,const int c2,const int d2)
{
    return (a2>=a1&&a2<=c1)||(c2>=a1&&c2<=c1)||
           (d2>=a1&&d2<=c1)||(b2>=a1&&b2<=c1)||
           (a2>=d1&&a2<=b1)||(c2>=d1&&c2<=b1)||
           (d2>=d1&&d2<=b1)||(b2>=d1&&b2<=b1);
}

/* find base pairing stacks in RNA*/
void sec_str(int len,char *seq, const vector<vector<bool> >&bp, 
    int a, int b,int &c, int &d)
{
    int i;
    
    for (i=0;i<len;i++)
    {
        if (a+i<len-3 && b-i>0)
        {
            if (a+i<b-i && bp[a+i][b-i]) continue;
            break;
        }
    }
    c=a+i-1;d=b-i+1;
}

/* secondary structure assignment for RNA:
 * 1->unpair, 2->paired with upstream, 3->paired with downstream */
void make_sec(char *seq, double **x, int len, char *sec,const string atom_opt)
{
    int ii,jj,i,j;

    float lb=12.5; // lower bound for " C3'"
    float ub=15.0; // upper bound for " C3'"
    if     (atom_opt==" C4'") {lb=14.0;ub=16.0;}
    else if(atom_opt==" C5'") {lb=16.0;ub=18.0;}
    else if(atom_opt==" O3'") {lb=13.5;ub=16.5;}
    else if(atom_opt==" O5'") {lb=15.5;ub=18.5;}
    else if(atom_opt==" P  ") {lb=16.5;ub=21.0;}

    float dis;
    vector<bool> bp_tmp(len,false);
    vector<vector<bool> > bp(len,bp_tmp);
    bp_tmp.clear();
    for (i=0; i<len; i++)
    {
        sec[i]='.';
        for (j=i+1; j<len; j++)
        {
            if (((seq[i]=='u'||seq[i]=='t')&&(seq[j]=='a'             ))||
                ((seq[i]=='a'             )&&(seq[j]=='u'||seq[j]=='t'))||
                ((seq[i]=='g'             )&&(seq[j]=='c'||seq[j]=='u'))||
                ((seq[i]=='c'||seq[i]=='u')&&(seq[j]=='g'             )))
            {
                dis=sqrt(dist(x[i], x[j]));
                bp[j][i]=bp[i][j]=(dis>lb && dis<ub);
            }
        }
    }
    
    // From 5' to 3': A0 C0 D0 B0: A0 paired to B0, C0 paired to D0
    vector<int> A0,B0,C0,D0;
    for (i=0; i<len-2; i++)
    {
        for (j=i+3; j<len; j++)
        {
            if (!bp[i][j]) continue;
            if (i>0 && j+1<len && bp[i-1][j+1]) continue;
            if (!bp[i+1][j-1]) continue;
            sec_str(len,seq, bp, i,j,ii,jj);
            if (jj<i || j<ii)
            {
                ii=i;
                jj=j;
            }
            A0.push_back(i);
            B0.push_back(j);
            C0.push_back(ii);
            D0.push_back(jj);
        }
    }
    
    //int sign;
    for (i=0;i<A0.size();i++)
    {
        /*
        sign=0;
        if(C0[i]-A0[i]<=1)
        {
            for(j=0;j<A0.size();j++)
            {
                if(i==j) continue;

                if((A0[j]>=A0[i]&&A0[j]<=C0[i])||
                   (C0[j]>=A0[i]&&C0[j]<=C0[i])||
                   (D0[j]>=A0[i]&&D0[j]<=C0[i])||
                   (B0[j]>=A0[i]&&B0[j]<=C0[i])||
                   (A0[j]>=D0[i]&&A0[j]<=B0[i])||
                   (C0[j]>=D0[i]&&C0[j]<=B0[i])||
                   (D0[j]>=D0[i]&&D0[j]<=B0[i])||
                   (B0[j]>=D0[i]&&B0[j]<=B0[i]))
                {
                    sign=-1;
                    break;
                }
            }
        }
        if(sign!=0) continue;
        */

        for (j=0;;j++)
        {
            if(A0[i]+j>C0[i]) break;
            sec[A0[i]+j]='<';
            sec[D0[i]+j]='>';
        }
    }
    sec[len]=0;

    /* clean up */
    A0.clear();
    B0.clear();
    C0.clear();
    D0.clear();
    bp.clear();
}

//get initial alignment from secondary structure alignment
//input: x, y, xlen, ylen
//output: y2x stores the best alignment: e.g., 
//y2x[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
void get_initial_ss(bool **path, double **val,
    const char *secx, const char *secy, int xlen, int ylen, int *y2x)
{
    double gap_open=-1.0;
    NWDP_TM(path, val, secx, secy, xlen, ylen, gap_open, y2x);
}


// get_initial5 in TMalign fortran, get_initial_local in TMalign c by yangji
//get initial alignment of local structure superposition
//input: x, y, xlen, ylen
//output: y2x stores the best alignment: e.g., 
//y2x[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
bool get_initial5( double **r1, double **r2, double **xtm, double **ytm,
    bool **path, double **val,
    double **x, double **y, int xlen, int ylen, int *y2x,
    double d0, double d0_search, const bool fast_opt, const double D0_MIN)
{
    double GL, rmsd;
    double t[3];
    double u[3][3];

    double d01 = d0 + 1.5;
    if (d01 < D0_MIN) d01 = D0_MIN;
    double d02 = d01*d01;

    double GLmax = 0;
    int aL = getmin(xlen, ylen);
    int *invmap = new int[ylen + 1];

    // jump on sequence1-------------->
    int n_jump1 = 0;
    if (xlen > 250)
        n_jump1 = 45;
    else if (xlen > 200)
        n_jump1 = 35;
    else if (xlen > 150)
        n_jump1 = 25;
    else
        n_jump1 = 15;
    if (n_jump1 > (xlen / 3))
        n_jump1 = xlen / 3;

    // jump on sequence2-------------->
    int n_jump2 = 0;
    if (ylen > 250)
        n_jump2 = 45;
    else if (ylen > 200)
        n_jump2 = 35;
    else if (ylen > 150)
        n_jump2 = 25;
    else
        n_jump2 = 15;
    if (n_jump2 > (ylen / 3))
        n_jump2 = ylen / 3;

    // fragment to superimpose-------------->
    int n_frag[2] = { 20, 100 };
    if (n_frag[0] > (aL / 3))
        n_frag[0] = aL / 3;
    if (n_frag[1] > (aL / 2))
        n_frag[1] = aL / 2;

    // start superimpose search-------------->
    if (fast_opt)
    {
        n_jump1*=5;
        n_jump2*=5;
    }
    bool flag = false;
    for (int i_frag = 0; i_frag < 2; i_frag++)
    {
        int m1 = xlen - n_frag[i_frag] + 1;
        int m2 = ylen - n_frag[i_frag] + 1;

        for (int i = 0; i<m1; i = i + n_jump1) //index starts from 0, different from FORTRAN
        {
            for (int j = 0; j<m2; j = j + n_jump2)
            {
                for (int k = 0; k<n_frag[i_frag]; k++) //fragment in y
                {
                    r1[k][0] = x[k + i][0];
                    r1[k][1] = x[k + i][1];
                    r1[k][2] = x[k + i][2];

                    r2[k][0] = y[k + j][0];
                    r2[k][1] = y[k + j][1];
                    r2[k][2] = y[k + j][2];
                }

                // superpose the two structures and rotate it
                Kabsch(r1, r2, n_frag[i_frag], 1, &rmsd, t, u);

                double gap_open = 0.0;
                NWDP_TM(path, val, x, y, xlen, ylen,
                    t, u, d02, gap_open, invmap);
                GL = get_score_fast(r1, r2, xtm, ytm, x, y, xlen, ylen,
                    invmap, d0, d0_search, t, u);
                if (GL>GLmax)
                {
                    GLmax = GL;
                    for (int ii = 0; ii<ylen; ii++) y2x[ii] = invmap[ii];
                    flag = true;
                }
            }
        }
    }

    delete[] invmap;
    return flag;
}

void score_matrix_rmsd_sec( double **r1, double **r2, double **score,
    const char *secx, const char *secy, double **x, double **y,
    int xlen, int ylen, int *y2x, const double D0_MIN, double d0)
{
    double t[3], u[3][3];
    double rmsd, dij;
    double d01=d0+1.5;
    if(d01 < D0_MIN) d01=D0_MIN;
    double d02=d01*d01;

    double xx[3];
    int i, k=0;
    for(int j=0; j<ylen; j++)
    {
        i=y2x[j];
        if(i>=0)
        {
            r1[k][0]=x[i][0];  
            r1[k][1]=x[i][1]; 
            r1[k][2]=x[i][2];   
            
            r2[k][0]=y[j][0];  
            r2[k][1]=y[j][1]; 
            r2[k][2]=y[j][2];
            
            k++;
        }
    }
    Kabsch(r1, r2, k, 1, &rmsd, t, u);

    
    for(int ii=0; ii<xlen; ii++)
    {        
        transform(t, u, &x[ii][0], xx);
        for(int jj=0; jj<ylen; jj++)
        {
            dij=dist(xx, &y[jj][0]); 
            if (secx[ii]==secy[jj])
                score[ii+1][jj+1] = 1.0/(1+dij/d02) + 0.5;
            else
                score[ii+1][jj+1] = 1.0/(1+dij/d02);
        }
    }
}


//get initial alignment from secondary structure and previous alignments
//input: x, y, xlen, ylen
//output: y2x stores the best alignment: e.g., 
//y2x[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
void get_initial_ssplus(double **r1, double **r2, double **score, bool **path,
    double **val, const char *secx, const char *secy, double **x, double **y,
    int xlen, int ylen, int *y2x0, int *y2x, const double D0_MIN, double d0)
{
    //create score matrix for DP
    score_matrix_rmsd_sec(r1, r2, score, secx, secy, x, y, xlen, ylen,
        y2x0, D0_MIN,d0);
    
    double gap_open=-1.0;
    NWDP_TM(score, path, val, xlen, ylen, gap_open, y2x);
}


void find_max_frag(double **x, int len, int *start_max,
    int *end_max, double dcu0, const bool fast_opt)
{
    int r_min, fra_min=4;           //minimum fragment for search
    if (fast_opt) fra_min=8;
    int start;
    int Lfr_max=0;

    r_min= (int) (len*1.0/3.0); //minimum fragment, in case too small protein
    if(r_min > fra_min) r_min=fra_min;
    
    int inc=0;
    double dcu0_cut=dcu0*dcu0;;
    double dcu_cut=dcu0_cut;

    while(Lfr_max < r_min)
    {        
        Lfr_max=0;            
        int j=1;    //number of residues at nf-fragment
        start=0;
        for(int i=1; i<len; i++)
        {
            if(dist(x[i-1], x[i]) < dcu_cut)
            {
                j++;

                if(i==(len-1))
                {
                    if(j > Lfr_max) 
                    {
                        Lfr_max=j;
                        *start_max=start;
                        *end_max=i;                        
                    }
                    j=1;
                }
            }
            else
            {
                if(j>Lfr_max) 
                {
                    Lfr_max=j;
                    *start_max=start;
                    *end_max=i-1;                                        
                }

                j=1;
                start=i;
            }
        }// for i;
        
        if(Lfr_max < r_min)
        {
            inc++;
            double dinc=pow(1.1, (double) inc) * dcu0;
            dcu_cut= dinc*dinc;
        }
    }//while <;    
}

//perform fragment gapless threading to find the best initial alignment
//input: x, y, xlen, ylen
//output: y2x0 stores the best alignment: e.g., 
//y2x0[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
double get_initial_fgt(double **r1, double **r2, double **xtm, double **ytm,
    double **x, double **y, int xlen, int ylen, 
    int *y2x, double d0, double d0_search,
    double dcu0, const bool fast_opt, double t[3], double u[3][3])
{
    int fra_min=4;           //minimum fragment for search
    if (fast_opt) fra_min=8;
    int fra_min1=fra_min-1;  //cutoff for shift, save time

    int xstart=0, ystart=0, xend=0, yend=0;

    find_max_frag(x, xlen, &xstart, &xend, dcu0, fast_opt);
    find_max_frag(y, ylen, &ystart, &yend, dcu0, fast_opt);


    int Lx = xend-xstart+1;
    int Ly = yend-ystart+1;
    int *ifr, *y2x_;
    int L_fr=getmin(Lx, Ly);
    ifr= new int[L_fr];
    y2x_= new int[ylen+1];

    //select what piece will be used. The original implement may cause 
    //asymetry, but only when xlen==ylen and Lx==Ly
    //if L1=Lfr1 and L2=Lfr2 (normal proteins), it will be the same as initial1

    if(Lx<Ly || (Lx==Ly && xlen<ylen))
    {        
        for(int i=0; i<L_fr; i++) ifr[i]=xstart+i;
    }
    else if(Lx>Ly || (Lx==Ly && xlen>ylen))
    {        
        for(int i=0; i<L_fr; i++) ifr[i]=ystart+i;
    }
    else // solve asymetric for 1x5gA vs 2q7nA5
    {
        /* In this case, L0==xlen==ylen; L_fr==Lx==Ly */
        int L0=xlen;
        double tmscore, tmscore_max=-1;
        int i, j, k;
        int n1, n2;
        int min_len;
        int min_ali;

        /* part 1, normalized by xlen */
        for(i=0; i<L_fr; i++) ifr[i]=xstart+i;

        if(L_fr==L0)
        {
            n1= (int)(L0*0.1); //my index starts from 0
            n2= (int)(L0*0.89);
            j=0;
            for(i=n1; i<= n2; i++)
            {
                ifr[j]=ifr[i];
                j++;
            }
            L_fr=j;
        }

        int L1=L_fr;
        min_len=getmin(L1, ylen);    
        min_ali= (int) (min_len/2.5); //minimum size of considered fragment 
        if(min_ali<=fra_min1)  min_ali=fra_min1;    
        n1 = -ylen+min_ali; 
        n2 = L1-min_ali;

        for(k=n1; k<=n2; k+=(fast_opt)?3:1)
        {
            //get the map
            for(j=0; j<ylen; j++)
            {
                i=j+k;
                if(i>=0 && i<L1) y2x_[j]=ifr[i];
                else             y2x_[j]=-1;
            }

            //evaluate the map quickly in three iterations
            tmscore=get_score_fast(r1, r2, xtm, ytm, x, y, xlen, ylen, y2x_,
                d0, d0_search, t, u);

            if(tmscore>=tmscore_max)
            {
                tmscore_max=tmscore;
                for(j=0; j<ylen; j++) y2x[j]=y2x_[j];
            }
        }

        /* part 2, normalized by ylen */
        L_fr=Ly;
        for(i=0; i<L_fr; i++) ifr[i]=ystart+i;

        if (L_fr==L0)
        {
            n1= (int)(L0*0.1); //my index starts from 0
            n2= (int)(L0*0.89);

            j=0;
            for(i=n1; i<= n2; i++)
            {
                ifr[j]=ifr[i];
                j++;
            }
            L_fr=j;
        }

        int L2=L_fr;
        min_len=getmin(xlen, L2);    
        min_ali= (int) (min_len/2.5); //minimum size of considered fragment 
        if(min_ali<=fra_min1)  min_ali=fra_min1;    
        n1 = -L2+min_ali; 
        n2 = xlen-min_ali;

        for(k=n1; k<=n2; k++)
        {
            //get the map
            for(j=0; j<ylen; j++) y2x_[j]=-1;

            for(j=0; j<L2; j++)
            {
                i=j+k;
                if(i>=0 && i<xlen) y2x_[ifr[j]]=i;
            }
        
            //evaluate the map quickly in three iterations
            tmscore=get_score_fast(r1, r2, xtm, ytm,
                x, y, xlen, ylen, y2x_, d0,d0_search, t, u);
            if(tmscore>=tmscore_max)
            {
                tmscore_max=tmscore;
                for(j=0; j<ylen; j++) y2x[j]=y2x_[j];
            }
        }

        delete [] ifr;
        delete [] y2x_;
        return tmscore_max;
    }

    
    int L0=getmin(xlen, ylen); //non-redundant to get_initial1
    if(L_fr==L0)
    {
        int n1= (int)(L0*0.1); //my index starts from 0
        int n2= (int)(L0*0.89);

        int j=0;
        for(int i=n1; i<= n2; i++)
        {
            ifr[j]=ifr[i];
            j++;
        }
        L_fr=j;
    }


    //gapless threading for the extracted fragment
    double tmscore, tmscore_max=-1;

    if(Lx<Ly || (Lx==Ly && xlen<=ylen))
    {
        int L1=L_fr;
        int min_len=getmin(L1, ylen);    
        int min_ali= (int) (min_len/2.5);              //minimum size of considered fragment 
        if(min_ali<=fra_min1)  min_ali=fra_min1;    
        int n1, n2;
        n1 = -ylen+min_ali; 
        n2 = L1-min_ali;

        int i, j, k;
        for(k=n1; k<=n2; k+=(fast_opt)?3:1)
        {
            //get the map
            for(j=0; j<ylen; j++)
            {
                i=j+k;
                if(i>=0 && i<L1) y2x_[j]=ifr[i];
                else             y2x_[j]=-1;
            }

            //evaluate the map quickly in three iterations
            tmscore=get_score_fast(r1, r2, xtm, ytm, x, y, xlen, ylen, y2x_,
                d0, d0_search, t, u);

            if(tmscore>=tmscore_max)
            {
                tmscore_max=tmscore;
                for(j=0; j<ylen; j++) y2x[j]=y2x_[j];
            }
        }
    }
    else
    {
        int L2=L_fr;
        int min_len=getmin(xlen, L2);    
        int min_ali= (int) (min_len/2.5);              //minimum size of considered fragment 
        if(min_ali<=fra_min1)  min_ali=fra_min1;    
        int n1, n2;
        n1 = -L2+min_ali; 
        n2 = xlen-min_ali;

        int i, j, k;    

        for(k=n1; k<=n2; k++)
        {
            //get the map
            for(j=0; j<ylen; j++) y2x_[j]=-1;

            for(j=0; j<L2; j++)
            {
                i=j+k;
                if(i>=0 && i<xlen) y2x_[ifr[j]]=i;
            }
        
            //evaluate the map quickly in three iterations
            tmscore=get_score_fast(r1, r2, xtm, ytm,
                x, y, xlen, ylen, y2x_, d0,d0_search, t, u);
            if(tmscore>=tmscore_max)
            {
                tmscore_max=tmscore;
                for(j=0; j<ylen; j++) y2x[j]=y2x_[j];
            }
        }
    }    


    delete [] ifr;
    delete [] y2x_;
    return tmscore_max;
}

//heuristic run of dynamic programing iteratively to find the best alignment
//input: initial rotation matrix t, u
//       vectors x and y, d0
//output: best alignment that maximizes the TMscore, will be stored in invmap
double DP_iter(double **r1, double **r2, double **xtm, double **ytm,
    double **xt, bool **path, double **val, double **x, double **y,
    int xlen, int ylen, double t[3], double u[3][3], int invmap0[],
    int g1, int g2, int iteration_max, double local_d0_search,
    double D0_MIN, double Lnorm, double d0, double score_d8)
{
    double gap_open[2]={-0.6, 0};
    double rmsd; 
    int *invmap=new int[ylen+1];
    
    int iteration, i, j, k;
    double tmscore, tmscore_max, tmscore_old=0;    
    int score_sum_method=8, simplify_step=40;
    tmscore_max=-1;

    //double d01=d0+1.5;
    double d02=d0*d0;
    for(int g=g1; g<g2; g++)
    {
        for(iteration=0; iteration<iteration_max; iteration++)
        {           
            NWDP_TM(path, val, x, y, xlen, ylen,
                t, u, d02, gap_open[g], invmap);
            
            k=0;
            for(j=0; j<ylen; j++) 
            {
                i=invmap[j];

                if(i>=0) //aligned
                {
                    xtm[k][0]=x[i][0];
                    xtm[k][1]=x[i][1];
                    xtm[k][2]=x[i][2];
                    
                    ytm[k][0]=y[j][0];
                    ytm[k][1]=y[j][1];
                    ytm[k][2]=y[j][2];
                    k++;
                }
            }

            tmscore = TMscore8_search(r1, r2, xtm, ytm, xt, k, t, u,
                simplify_step, score_sum_method, &rmsd, local_d0_search,
                Lnorm, score_d8, d0);

           
            if(tmscore>tmscore_max)
            {
                tmscore_max=tmscore;
                for(i=0; i<ylen; i++) invmap0[i]=invmap[i];
            }
    
            if(iteration>0)
            {
                if(fabs(tmscore_old-tmscore)<0.000001) break;       
            }
            tmscore_old=tmscore;
        }// for iteration           
        
    }//for gapopen
    
    
    delete []invmap;
    return tmscore_max;
}


void output_pymol(const string xname, const string yname,
    const string fname_super, double t[3], double u[3][3], const int ter_opt, 
    const int mm_opt, const int split_opt, const int mirror_opt,
    const char *seqM, const char *seqxA, const char *seqyA,
    const vector<string>&resi_vec1, const vector<string>&resi_vec2,
    const string chainID1, const string chainID2)
{
    int compress_type=0; // uncompressed file
    ifstream fin;
#ifndef REDI_PSTREAM_H_SEEN
    ifstream fin_gz;
#else
    redi::ipstream fin_gz; // if file is compressed
    if (xname.size()>=3 && 
        xname.substr(xname.size()-3,3)==".gz")
    {
        fin_gz.open("gunzip -c "+xname);
        compress_type=1;
    }
    else if (xname.size()>=4 && 
        xname.substr(xname.size()-4,4)==".bz2")
    {
        fin_gz.open("bzcat "+xname);
        compress_type=2;
    }
    else
#endif
        fin.open(xname.c_str());

    stringstream buf;
    stringstream buf_pymol;
    string line;
    double x[3];  // before transform
    double x1[3]; // after transform

    /* for PDBx/mmCIF only */
    map<string,int> _atom_site;
    size_t atom_site_pos;
    vector<string> line_vec;
    int infmt=-1; // 0 - PDB, 3 - PDBx/mmCIF

    while (compress_type?fin_gz.good():fin.good())
    {
        if (compress_type) getline(fin_gz, line);
        else               getline(fin, line);
        if (line.compare(0, 6, "ATOM  ")==0 || 
            line.compare(0, 6, "HETATM")==0) // PDB format
        {
            infmt=0;
            x[0]=atof(line.substr(30,8).c_str());
            x[1]=atof(line.substr(38,8).c_str());
            x[2]=atof(line.substr(46,8).c_str());
            if (mirror_opt) x[2]=-x[2];
            transform(t, u, x, x1);
            buf<<line.substr(0,30)<<setiosflags(ios::fixed)
                <<setprecision(3)
                <<setw(8)<<x1[0] <<setw(8)<<x1[1] <<setw(8)<<x1[2]
                <<line.substr(54)<<'\n';
        }
        else if (line.compare(0,5,"loop_")==0) // PDBx/mmCIF
        {
            infmt=3;
            buf<<line<<'\n';
            while(1)
            {
                if (compress_type) 
                {
                    if (fin_gz.good()) getline(fin_gz, line);
                    else PrintErrorAndQuit("ERROR! Unexpected end of "+xname);
                }
                else
                {
                    if (fin.good()) getline(fin, line);
                    else PrintErrorAndQuit("ERROR! Unexpected end of "+xname);
                }
                if (line.size()) break;
            }
            buf<<line<<'\n';
            if (line.compare(0,11,"_atom_site.")) continue;
            _atom_site.clear();
            atom_site_pos=0;
            _atom_site[Trim(line.substr(11))]=atom_site_pos;
            while(1)
            {
                while(1)
                {
                    if (compress_type) 
                    {
                        if (fin_gz.good()) getline(fin_gz, line);
                        else PrintErrorAndQuit("ERROR! Unexpected end of "+xname);
                    }
                    else
                    {
                        if (fin.good()) getline(fin, line);
                        else PrintErrorAndQuit("ERROR! Unexpected end of "+xname);
                    }
                    if (line.size()) break;
                }
                if (line.compare(0,11,"_atom_site.")) break;
                _atom_site[Trim(line.substr(11))]=++atom_site_pos;
                buf<<line<<'\n';
            }

            if (_atom_site.count("group_PDB")*
                _atom_site.count("Cartn_x")*
                _atom_site.count("Cartn_y")*
                _atom_site.count("Cartn_z")==0)
            {
                buf<<line<<'\n';
                cerr<<"Warning! Missing one of the following _atom_site data items: group_PDB, Cartn_x, Cartn_y, Cartn_z"<<endl;
                continue;
            }

            while(1)
            {
                line_vec.clear();
                split(line,line_vec);
                if (line_vec[_atom_site["group_PDB"]]!="ATOM" &&
                    line_vec[_atom_site["group_PDB"]]!="HETATM") break;

                x[0]=atof(line_vec[_atom_site["Cartn_x"]].c_str());
                x[1]=atof(line_vec[_atom_site["Cartn_y"]].c_str());
                x[2]=atof(line_vec[_atom_site["Cartn_z"]].c_str());
                if (mirror_opt) x[2]=-x[2];
                transform(t, u, x, x1);

                for (atom_site_pos=0; atom_site_pos<_atom_site.size(); atom_site_pos++)
                {
                    if (atom_site_pos==_atom_site["Cartn_x"])
                        buf<<setiosflags(ios::fixed)<<setprecision(3)
                           <<setw(8)<<x1[0]<<' ';
                    else if (atom_site_pos==_atom_site["Cartn_y"])
                        buf<<setiosflags(ios::fixed)<<setprecision(3)
                           <<setw(8)<<x1[1]<<' ';
                    else if (atom_site_pos==_atom_site["Cartn_z"])
                        buf<<setiosflags(ios::fixed)<<setprecision(3)
                           <<setw(8)<<x1[2]<<' ';
                    else buf<<line_vec[atom_site_pos]<<' ';
                }
                buf<<'\n';

                if (compress_type && fin_gz.good()) getline(fin_gz, line);
                else if (!compress_type && fin.good()) getline(fin, line);
                else break;
            }
            if (compress_type?fin_gz.good():fin.good()) buf<<line<<'\n';
        }
        else if (line.size())
        {
            buf<<line<<'\n';
            if (ter_opt>=1 && line.compare(0,3,"END")==0) break;
        }
    }
    if (compress_type) fin_gz.close();
    else               fin.close();

    string fname_super_full=fname_super;
    if (infmt==0)      fname_super_full+=".pdb";
    else if (infmt==3) fname_super_full+=".cif";
    ofstream fp;
    fp.open(fname_super_full.c_str());
    fp<<buf.str();
    fp.close();
    buf.str(string()); // clear stream

    string chain1_sele;
    string chain2_sele;
    int i;
    if (!mm_opt)
    {
        if (split_opt==2 && ter_opt>=1) // align one chain from model 1
        {
            chain1_sele=" and c. "+chainID1.substr(1);
            chain2_sele=" and c. "+chainID2.substr(1);
        }
        else if (split_opt==2 && ter_opt==0) // align one chain from each model
        {
            for (i=1;i<chainID1.size();i++) if (chainID1[i]==',') break;
            chain1_sele=" and c. "+chainID1.substr(i+1);
            for (i=1;i<chainID2.size();i++) if (chainID2[i]==',') break;
            chain2_sele=" and c. "+chainID2.substr(i+1);
        }
    }

    /* extract aligned region */
    int i1=-1;
    int i2=-1;
    string resi1_sele;
    string resi2_sele;
    string resi1_bond;
    string resi2_bond;
    string prev_resi1;
    string prev_resi2;
    string curr_resi1;
    string curr_resi2;
    if (mm_opt)
    {
        ;
    }
    else
    {
        for (i=0;i<strlen(seqM);i++)
        {
            i1+=(seqxA[i]!='-' && seqxA[i]!='*');
            i2+=(seqyA[i]!='-');
            if (seqM[i]==' ' || seqxA[i]=='*') continue;
            curr_resi1=resi_vec1[i1].substr(0,4);
            curr_resi2=resi_vec2[i2].substr(0,4);
            if (resi1_sele.size()==0)
                resi1_sele =    "i. "+curr_resi1;
            else
            {
                resi1_sele+=" or i. "+curr_resi1;
                resi1_bond+="bond structure1 and i. "+prev_resi1+
                                              ", i. "+curr_resi1+"\n";
            }
            if (resi2_sele.size()==0)
                resi2_sele =    "i. "+curr_resi2;
            else
            {
                resi2_sele+=" or i. "+curr_resi2;
                resi2_bond+="bond structure2 and i. "+prev_resi2+
                                              ", i. "+curr_resi2+"\n";
            }
            prev_resi1=curr_resi1;
            prev_resi2=curr_resi2;
            //if (seqM[i]!=':') continue;
        }
        if (resi1_sele.size()) resi1_sele=" and ( "+resi1_sele+")";
        if (resi2_sele.size()) resi2_sele=" and ( "+resi2_sele+")";
    }

    /* write pymol script */
    vector<string> pml_list;
    pml_list.push_back(fname_super+"");
    pml_list.push_back(fname_super+"_atm");
    pml_list.push_back(fname_super+"_all");
    pml_list.push_back(fname_super+"_all_atm");
    pml_list.push_back(fname_super+"_all_atm_lig");

    for (int p=0;p<pml_list.size();p++)
    {
        if (mm_opt && p<=1) continue;
        buf_pymol
            <<"#!/usr/bin/env pymol\n"
            <<"cmd.load(\""<<fname_super_full<<"\", \"structure1\")\n"
            <<"cmd.load(\""<<yname<<"\", \"structure2\")\n"
            <<"hide all\n"
            <<"set all_states, "<<((ter_opt==0)?"on":"off")<<'\n';
        if (p==0) // .pml
        {
            if (chain1_sele.size()) buf_pymol
                <<"remove structure1 and not "<<chain1_sele.substr(4)<<"\n";
            if (chain2_sele.size()) buf_pymol
                <<"remove structure2 and not "<<chain2_sele.substr(4)<<"\n";
            buf_pymol
                <<"remove not n. CA and not n. C3'\n"
                <<resi1_bond
                <<resi2_bond
                <<"show stick, structure1"<<chain1_sele<<resi1_sele<<"\n"
                <<"show stick, structure2"<<chain2_sele<<resi2_sele<<"\n";
        }
        else if (p==1) // _atm.pml
        {
            buf_pymol
                <<"show cartoon, structure1"<<chain1_sele<<resi1_sele<<"\n"
                <<"show cartoon, structure2"<<chain2_sele<<resi2_sele<<"\n";
        }
        else if (p==2) // _all.pml
        {
            buf_pymol
                <<"show ribbon, structure1"<<chain1_sele<<"\n"
                <<"show ribbon, structure2"<<chain2_sele<<"\n";
        }
        else if (p==3) // _all_atm.pml
        {
            buf_pymol
                <<"show cartoon, structure1"<<chain1_sele<<"\n"
                <<"show cartoon, structure2"<<chain2_sele<<"\n";
        }
        else if (p==4) // _all_atm_lig.pml
        {
            buf_pymol
                <<"show cartoon, structure1\n"
                <<"show cartoon, structure2\n"
                <<"show stick, not polymer\n"
                <<"show sphere, not polymer\n";
        }
        buf_pymol
            <<"color blue, structure1\n"
            <<"color red, structure2\n"
            <<"set ribbon_width, 6\n"
            <<"set stick_radius, 0.3\n"
            <<"set sphere_scale, 0.25\n"
            <<"set ray_shadow, 0\n"
            <<"bg_color white\n"
            <<"set transparency=0.2\n"
            <<"zoom polymer and ((structure1"<<chain1_sele
            <<") or (structure2"<<chain2_sele<<"))\n"
            <<endl;

        fp.open((pml_list[p]+".pml").c_str());
        fp<<buf_pymol.str();
        fp.close();
        buf_pymol.str(string());
    }

    /* clean up */
    pml_list.clear();
    
    resi1_sele.clear();
    resi2_sele.clear();
    
    resi1_bond.clear();
    resi2_bond.clear();
    
    prev_resi1.clear();
    prev_resi2.clear();

    curr_resi1.clear();
    curr_resi2.clear();

    chain1_sele.clear();
    chain2_sele.clear();
}

void output_rasmol(const string xname, const string yname,
    const string fname_super, double t[3], double u[3][3], const int ter_opt,
    const int mm_opt, const int split_opt, const int mirror_opt,
    const char *seqM, const char *seqxA, const char *seqyA,
    const vector<string>&resi_vec1, const vector<string>&resi_vec2,
    const string chainID1, const string chainID2,
    const int xlen, const int ylen, const double d0A, const int n_ali8,
    const double rmsd, const double TM1, const double Liden)
{
    stringstream buf;
    stringstream buf_all;
    stringstream buf_atm;
    stringstream buf_all_atm;
    stringstream buf_all_atm_lig;
    //stringstream buf_pdb;
    stringstream buf_tm;
    string line;
    double x[3];  // before transform
    double x1[3]; // after transform
    bool after_ter; // true if passed the "TER" line in PDB
    string asym_id; // chain ID

    buf_tm<<"REMARK US-align"
        <<"\nREMARK Structure 1:"<<setw(11)<<left<<xname+chainID1<<" Size= "<<xlen
        <<"\nREMARK Structure 2:"<<setw(11)<<yname+chainID2<<right<<" Size= "<<ylen
        <<" (TM-score is normalized by "<<setw(4)<<ylen<<", d0="
        <<setiosflags(ios::fixed)<<setprecision(2)<<setw(6)<<d0A<<")"
        <<"\nREMARK Aligned length="<<setw(4)<<n_ali8<<", RMSD="
        <<setw(6)<<setiosflags(ios::fixed)<<setprecision(2)<<rmsd
        <<", TM-score="<<setw(7)<<setiosflags(ios::fixed)<<setprecision(5)<<TM1
        <<", ID="<<setw(5)<<setiosflags(ios::fixed)<<setprecision(3)
        <<((n_ali8>0)?Liden/n_ali8:0)<<endl;
    string rasmol_CA_header="load inline\nselect *A\nwireframe .45\nselect *B\nwireframe .20\nselect all\ncolor white\n";
    string rasmol_cartoon_header="load inline\nselect all\ncartoon\nselect *A\ncolor blue\nselect *B\ncolor red\nselect ligand\nwireframe 0.25\nselect solvent\nspacefill 0.25\nselect all\nexit\n"+buf_tm.str();
    if (!mm_opt) buf<<rasmol_CA_header;
    buf_all<<rasmol_CA_header;
    if (!mm_opt) buf_atm<<rasmol_cartoon_header;
    buf_all_atm<<rasmol_cartoon_header;
    buf_all_atm_lig<<rasmol_cartoon_header;

    /* selecting chains for -mol */
    string chain1_sele;
    string chain2_sele;
    int i;
    if (!mm_opt)
    {
        if (split_opt==2 && ter_opt>=1) // align one chain from model 1
        {
            chain1_sele=chainID1.substr(1);
            chain2_sele=chainID2.substr(1);
        }
        else if (split_opt==2 && ter_opt==0) // align one chain from each model
        {
            for (i=1;i<chainID1.size();i++) if (chainID1[i]==',') break;
            chain1_sele=chainID1.substr(i+1);
            for (i=1;i<chainID2.size();i++) if (chainID2[i]==',') break;
            chain2_sele=chainID2.substr(i+1);
        }
    }


    /* for PDBx/mmCIF only */
    map<string,int> _atom_site;
    int atom_site_pos;
    vector<string> line_vec;
    string atom; // 4-character atom name
    string AA;   // 3-character residue name
    string resi; // 4-character residue sequence number
    string inscode; // 1-character insertion code
    string model_index; // model index
    bool is_mmcif=false;

    /* used for CONECT record of chain1 */
    int ca_idx1=0; // all CA atoms
    int lig_idx1=0; // all atoms
    vector <int> idx_vec;

    /* used for CONECT record of chain2 */
    int ca_idx2=0; // all CA atoms
    int lig_idx2=0; // all atoms

    /* extract aligned region */
    vector<string> resi_aln1;
    vector<string> resi_aln2;
    int i1=-1;
    int i2=-1;
    if (!mm_opt)
    {
        for (i=0;i<strlen(seqM);i++)
        {
            i1+=(seqxA[i]!='-');
            i2+=(seqyA[i]!='-');
            if (seqM[i]==' ') continue;
            resi_aln1.push_back(resi_vec1[i1].substr(0,4));
            resi_aln2.push_back(resi_vec2[i2].substr(0,4));
            if (seqM[i]!=':') continue;
            buf    <<"select "<<resi_aln1.back()<<":A,"
                   <<resi_aln2.back()<<":B\ncolor red\n";
            buf_all<<"select "<<resi_aln1.back()<<":A,"
                   <<resi_aln2.back()<<":B\ncolor red\n";
        }
        buf<<"select all\nexit\n"<<buf_tm.str();
    }
    buf_all<<"select all\nexit\n"<<buf_tm.str();

    ifstream fin;
    /* read first file */
    after_ter=false;
    asym_id="";
    fin.open(xname.c_str());
    while (fin.good())
    {
        getline(fin, line);
        if (ter_opt>=3 && line.compare(0,3,"TER")==0) after_ter=true;
        if (is_mmcif==false && line.size()>=54 &&
           (line.compare(0, 6, "ATOM  ")==0 ||
            line.compare(0, 6, "HETATM")==0)) // PDB format
        {
            if (line[16]!='A' && line[16]!=' ') continue;
            x[0]=atof(line.substr(30,8).c_str());
            x[1]=atof(line.substr(38,8).c_str());
            x[2]=atof(line.substr(46,8).c_str());
            if (mirror_opt) x[2]=-x[2];
            transform(t, u, x, x1);
            //buf_pdb<<line.substr(0,30)<<setiosflags(ios::fixed)
                //<<setprecision(3)
                //<<setw(8)<<x1[0] <<setw(8)<<x1[1] <<setw(8)<<x1[2]
                //<<line.substr(54)<<'\n';

            if (after_ter && line.compare(0,6,"ATOM  ")==0) continue;
            lig_idx1++;
            buf_all_atm_lig<<line.substr(0,6)<<setw(5)<<lig_idx1
                <<line.substr(11,9)<<" A"<<line.substr(22,8)
                <<setiosflags(ios::fixed)<<setprecision(3)
                <<setw(8)<<x1[0]<<setw(8)<<x1[1] <<setw(8)<<x1[2]<<'\n';
            if (chain1_sele.size() && line[21]!=chain1_sele[0]) continue;
            if (after_ter || line.compare(0,6,"ATOM  ")) continue;
            if (ter_opt>=2)
            {
                if (ca_idx1 && asym_id.size() && asym_id!=line.substr(21,1)) 
                {
                    after_ter=true;
                    continue;
                }
                asym_id=line[21];
            }
            buf_all_atm<<"ATOM  "<<setw(5)<<lig_idx1
                <<line.substr(11,9)<<" A"<<line.substr(22,8)
                <<setiosflags(ios::fixed)<<setprecision(3)
                <<setw(8)<<x1[0]<<setw(8)<<x1[1] <<setw(8)<<x1[2]<<'\n';
            if (!mm_opt && find(resi_aln1.begin(),resi_aln1.end(),
                line.substr(22,4))!=resi_aln1.end())
            {
                buf_atm<<"ATOM  "<<setw(5)<<lig_idx1
                    <<line.substr(11,9)<<" A"<<line.substr(22,8)
                    <<setiosflags(ios::fixed)<<setprecision(3)
                    <<setw(8)<<x1[0]<<setw(8)<<x1[1] <<setw(8)<<x1[2]<<'\n';
            }
            if (line.substr(12,4)!=" CA " && line.substr(12,4)!=" C3'") continue;
            ca_idx1++;
            buf_all<<"ATOM  "<<setw(5)<<ca_idx1<<' '
                <<line.substr(12,4)<<' '<<line.substr(17,3)<<" A"<<line.substr(22,8)
                <<setiosflags(ios::fixed)<<setprecision(3)
                <<setw(8)<<x1[0]<<setw(8)<<x1[1]<<setw(8)<<x1[2]<<'\n';
            if (find(resi_aln1.begin(),resi_aln1.end(),
                line.substr(22,4))==resi_aln1.end()) continue;
            if (!mm_opt) buf<<"ATOM  "<<setw(5)<<ca_idx1<<' '
                <<line.substr(12,4)<<' '<<line.substr(17,3)<<" A"<<line.substr(22,8)
                <<setiosflags(ios::fixed)<<setprecision(3)
                <<setw(8)<<x1[0]<<setw(8)<<x1[1]<<setw(8)<<x1[2]<<'\n';
            idx_vec.push_back(ca_idx1);
        }
        else if (line.compare(0,5,"loop_")==0) // PDBx/mmCIF
        {
            while(1)
            {
                if (fin.good()) getline(fin, line);
                else PrintErrorAndQuit("ERROR! Unexpected end of "+xname);
                if (line.size()) break;
            }
            if (line.compare(0,11,"_atom_site.")) continue;
            _atom_site.clear();
            atom_site_pos=0;
            _atom_site[line.substr(11,line.size()-12)]=atom_site_pos;
            while(1)
            {
                if (fin.good()) getline(fin, line);
                else PrintErrorAndQuit("ERROR! Unexpected end of "+xname);
                if (line.size()==0) continue;
                if (line.compare(0,11,"_atom_site.")) break;
                _atom_site[line.substr(11,line.size()-12)]=++atom_site_pos;
            }

            if (is_mmcif==false)
            {
                //buf_pdb.str(string());
                is_mmcif=true;
            }

            while(1)
            {
                line_vec.clear();
                split(line,line_vec);
                if (line_vec[_atom_site["group_PDB"]]!="ATOM" &&
                    line_vec[_atom_site["group_PDB"]]!="HETATM") break;
                if (_atom_site.count("pdbx_PDB_model_num"))
                {
                    if (model_index.size() && model_index!=
                        line_vec[_atom_site["pdbx_PDB_model_num"]])
                        break;
                    model_index=line_vec[_atom_site["pdbx_PDB_model_num"]];
                }

                x[0]=atof(line_vec[_atom_site["Cartn_x"]].c_str());
                x[1]=atof(line_vec[_atom_site["Cartn_y"]].c_str());
                x[2]=atof(line_vec[_atom_site["Cartn_z"]].c_str());
                if (mirror_opt) x[2]=-x[2];
                transform(t, u, x, x1);

                if (_atom_site.count("label_alt_id")==0 || 
                    line_vec[_atom_site["label_alt_id"]]=="." ||
                    line_vec[_atom_site["label_alt_id"]]=="A")
                {
                    atom=line_vec[_atom_site["label_atom_id"]];
                    if (atom[0]=='"') atom=atom.substr(1);
                    if (atom.size() && atom[atom.size()-1]=='"')
                        atom=atom.substr(0,atom.size()-1);
                    if      (atom.size()==0) atom="    ";
                    else if (atom.size()==1) atom=" "+atom+"  ";
                    else if (atom.size()==2) atom=" "+atom+" ";
                    else if (atom.size()==3) atom=" "+atom;
                    else if (atom.size()>=5) atom=atom.substr(0,4);
            
                    AA=line_vec[_atom_site["label_comp_id"]]; // residue name
                    if      (AA.size()==1) AA="  "+AA;
                    else if (AA.size()==2) AA=" " +AA;
                    else if (AA.size()>=4) AA=AA.substr(0,3);
                
                    if (_atom_site.count("auth_seq_id"))
                        resi=line_vec[_atom_site["auth_seq_id"]];
                    else resi=line_vec[_atom_site["label_seq_id"]];
                    while (resi.size()<4) resi=' '+resi;
                    if (resi.size()>4) resi=resi.substr(0,4);
                
                    inscode=' ';
                    if (_atom_site.count("pdbx_PDB_ins_code") && 
                        line_vec[_atom_site["pdbx_PDB_ins_code"]]!="?")
                        inscode=line_vec[_atom_site["pdbx_PDB_ins_code"]][0];

                    if (_atom_site.count("auth_asym_id"))
                    {
                        if (chain1_sele.size()) after_ter
                            =line_vec[_atom_site["auth_asym_id"]]!=chain1_sele;
                        else if (ter_opt>=2 && ca_idx1 && asym_id.size() && 
                            asym_id!=line_vec[_atom_site["auth_asym_id"]])
                            after_ter=true;
                        asym_id=line_vec[_atom_site["auth_asym_id"]];
                    }
                    else if (_atom_site.count("label_asym_id"))
                    {
                        if (chain1_sele.size()) after_ter
                            =line_vec[_atom_site["label_asym_id"]]!=chain1_sele;
                        if (ter_opt>=2 && ca_idx1 && asym_id.size() && 
                            asym_id!=line_vec[_atom_site["label_asym_id"]])
                            after_ter=true;
                        asym_id=line_vec[_atom_site["label_asym_id"]];
                    }
                    //buf_pdb<<left<<setw(6)
                        //<<line_vec[_atom_site["group_PDB"]]<<right
                        //<<setw(5)<<lig_idx1%100000<<' '<<atom<<' '
                        //<<AA<<" "<<asym_id[asym_id.size()-1]
                        //<<resi<<inscode<<"   "
                        //<<setiosflags(ios::fixed)<<setprecision(3)
                        //<<setw(8)<<x1[0]
                        //<<setw(8)<<x1[1]
                        //<<setw(8)<<x1[2]<<'\n';

                    if (after_ter==false ||
                        line_vec[_atom_site["group_pdb"]]=="HETATM")
                    {
                        lig_idx1++;
                        buf_all_atm_lig<<left<<setw(6)
                            <<line_vec[_atom_site["group_PDB"]]<<right
                            <<setw(5)<<lig_idx1%100000<<' '<<atom<<' '
                            <<AA<<" A"<<resi<<inscode<<"   "
                            <<setiosflags(ios::fixed)<<setprecision(3)
                            <<setw(8)<<x1[0]
                            <<setw(8)<<x1[1]
                            <<setw(8)<<x1[2]<<'\n';
                        if (after_ter==false &&
                            line_vec[_atom_site["group_PDB"]]=="ATOM")
                        {
                            buf_all_atm<<"ATOM  "<<setw(6)
                                <<setw(5)<<lig_idx1%100000<<' '<<atom<<' '
                                <<AA<<" A"<<resi<<inscode<<"   "
                                <<setiosflags(ios::fixed)<<setprecision(3)
                                <<setw(8)<<x1[0]
                                <<setw(8)<<x1[1]
                                <<setw(8)<<x1[2]<<'\n';
                            if (!mm_opt && find(resi_aln1.begin(),
                                resi_aln1.end(),resi)!=resi_aln1.end())
                            {
                                buf_atm<<"ATOM  "<<setw(6)
                                    <<setw(5)<<lig_idx1%100000<<' '
                                    <<atom<<' '<<AA<<" A"<<resi<<inscode<<"   "
                                    <<setiosflags(ios::fixed)<<setprecision(3)
                                    <<setw(8)<<x1[0]
                                    <<setw(8)<<x1[1]
                                    <<setw(8)<<x1[2]<<'\n';
                            }
                            if (atom==" CA " || atom==" C3'")
                            {
                                ca_idx1++;
            //mm_opt, split_opt, mirror_opt, chainID1,chainID2);
                                buf_all<<"ATOM  "<<setw(6)
                                    <<setw(5)<<ca_idx1%100000<<' '<<atom<<' '
                                    <<AA<<" A"<<resi<<inscode<<"   "
                                    <<setiosflags(ios::fixed)<<setprecision(3)
                                    <<setw(8)<<x1[0]
                                    <<setw(8)<<x1[1]
                                    <<setw(8)<<x1[2]<<'\n';
                                if (!mm_opt && find(resi_aln1.begin(),
                                    resi_aln1.end(),resi)!=resi_aln1.end())
                                {
                                    buf<<"ATOM  "<<setw(6)
                                    <<setw(5)<<ca_idx1%100000<<' '<<atom<<' '
                                    <<AA<<" A"<<resi<<inscode<<"   "
                                    <<setiosflags(ios::fixed)<<setprecision(3)
                                    <<setw(8)<<x1[0]
                                    <<setw(8)<<x1[1]
                                    <<setw(8)<<x1[2]<<'\n';
                                    idx_vec.push_back(ca_idx1);
                                }
                            }
                        }
                    }
                }

                while(1)
                {
                    if (fin.good()) getline(fin, line);
                    else break;
                    if (line.size()) break;
                }
            }
        }
        else if (line.size() && is_mmcif==false)
        {
            //buf_pdb<<line<<'\n';
            if (ter_opt>=1 && line.compare(0,3,"END")==0) break;
        }
    }
    fin.close();
    if (!mm_opt) buf<<"TER\n";
    buf_all<<"TER\n";
    if (!mm_opt) buf_atm<<"TER\n";
    buf_all_atm<<"TER\n";
    buf_all_atm_lig<<"TER\n";
    for (i=1;i<ca_idx1;i++) buf_all<<"CONECT"
        <<setw(5)<<i%100000<<setw(5)<<(i+1)%100000<<'\n';
    if (!mm_opt) for (i=1;i<idx_vec.size();i++) buf<<"CONECT"
        <<setw(5)<<idx_vec[i-1]%100000<<setw(5)<<idx_vec[i]%100000<<'\n';
    idx_vec.clear();

    /* read second file */
    after_ter=false;
    asym_id="";
    fin.open(yname.c_str());
    while (fin.good())
    {
        getline(fin, line);
        if (ter_opt>=3 && line.compare(0,3,"TER")==0) after_ter=true;
        if (line.size()>=54 && (line.compare(0, 6, "ATOM  ")==0 ||
            line.compare(0, 6, "HETATM")==0)) // PDB format
        {
            if (line[16]!='A' && line[16]!=' ') continue;
            if (after_ter && line.compare(0,6,"ATOM  ")==0) continue;
            lig_idx2++;
            buf_all_atm_lig<<line.substr(0,6)<<setw(5)<<lig_idx1+lig_idx2
                <<line.substr(11,9)<<" B"<<line.substr(22,32)<<'\n';
            if (chain1_sele.size() && line[21]!=chain1_sele[0]) continue;
            if (after_ter || line.compare(0,6,"ATOM  ")) continue;
            if (ter_opt>=2)
            {
                if (ca_idx2 && asym_id.size() && asym_id!=line.substr(21,1))
                {
                    after_ter=true;
                    continue;
                }
                asym_id=line[21];
            }
            buf_all_atm<<"ATOM  "<<setw(5)<<lig_idx1+lig_idx2
                <<line.substr(11,9)<<" B"<<line.substr(22,32)<<'\n';
            if (!mm_opt && find(resi_aln2.begin(),resi_aln2.end(),
                line.substr(22,4))!=resi_aln2.end())
            {
                buf_atm<<"ATOM  "<<setw(5)<<lig_idx1+lig_idx2
                    <<line.substr(11,9)<<" B"<<line.substr(22,32)<<'\n';
            }
            if (line.substr(12,4)!=" CA " && line.substr(12,4)!=" C3'") continue;
            ca_idx2++;
            buf_all<<"ATOM  "<<setw(5)<<ca_idx1+ca_idx2<<' '<<line.substr(12,4)
                <<' '<<line.substr(17,3)<<" B"<<line.substr(22,32)<<'\n';
            if (find(resi_aln2.begin(),resi_aln2.end(),line.substr(22,4)
                )==resi_aln2.end()) continue;
            if (!mm_opt) buf<<"ATOM  "<<setw(5)<<ca_idx1+ca_idx2<<' '
                <<line.substr(12,4)<<' '<<line.substr(17,3)<<" B"
                <<line.substr(22,32)<<'\n';
            idx_vec.push_back(ca_idx1+ca_idx2);
        }
        else if (line.compare(0,5,"loop_")==0) // PDBx/mmCIF
        {
            while(1)
            {
                if (fin.good()) getline(fin, line);
                else PrintErrorAndQuit("ERROR! Unexpected end of "+yname);
                if (line.size()) break;
            }
            if (line.compare(0,11,"_atom_site.")) continue;
            _atom_site.clear();
            atom_site_pos=0;
            _atom_site[line.substr(11,line.size()-12)]=atom_site_pos;
            while(1)
            {
                if (fin.good()) getline(fin, line);
                else PrintErrorAndQuit("ERROR! Unexpected end of "+yname);
                if (line.size()==0) continue;
                if (line.compare(0,11,"_atom_site.")) break;
                _atom_site[line.substr(11,line.size()-12)]=++atom_site_pos;
            }

            while(1)
            {
                line_vec.clear();
                split(line,line_vec);
                if (line_vec[_atom_site["group_PDB"]]!="ATOM" &&
                    line_vec[_atom_site["group_PDB"]]!="HETATM") break;
                if (_atom_site.count("pdbx_PDB_model_num"))
                {
                    if (model_index.size() && model_index!=
                        line_vec[_atom_site["pdbx_PDB_model_num"]])
                        break;
                    model_index=line_vec[_atom_site["pdbx_PDB_model_num"]];
                }

                if (_atom_site.count("label_alt_id")==0 || 
                    line_vec[_atom_site["label_alt_id"]]=="." ||
                    line_vec[_atom_site["label_alt_id"]]=="A")
                {
                    atom=line_vec[_atom_site["label_atom_id"]];
                    if (atom[0]=='"') atom=atom.substr(1);
                    if (atom.size() && atom[atom.size()-1]=='"')
                        atom=atom.substr(0,atom.size()-1);
                    if      (atom.size()==0) atom="    ";
                    else if (atom.size()==1) atom=" "+atom+"  ";
                    else if (atom.size()==2) atom=" "+atom+" ";
                    else if (atom.size()==3) atom=" "+atom;
                    else if (atom.size()>=5) atom=atom.substr(0,4);
            
                    AA=line_vec[_atom_site["label_comp_id"]]; // residue name
                    if      (AA.size()==1) AA="  "+AA;
                    else if (AA.size()==2) AA=" " +AA;
                    else if (AA.size()>=4) AA=AA.substr(0,3);
                
                    if (_atom_site.count("auth_seq_id"))
                        resi=line_vec[_atom_site["auth_seq_id"]];
                    else resi=line_vec[_atom_site["label_seq_id"]];
                    while (resi.size()<4) resi=' '+resi;
                    if (resi.size()>4) resi=resi.substr(0,4);
                
                    inscode=' ';
                    if (_atom_site.count("pdbx_PDB_ins_code") && 
                        line_vec[_atom_site["pdbx_PDB_ins_code"]]!="?")
                        inscode=line_vec[_atom_site["pdbx_PDB_ins_code"]][0];
                    
                    if (_atom_site.count("auth_asym_id"))
                    {
                        if (chain2_sele.size()) after_ter
                            =line_vec[_atom_site["auth_asym_id"]]!=chain2_sele;
                        if (ter_opt>=2 && ca_idx2 && asym_id.size() && 
                            asym_id!=line_vec[_atom_site["auth_asym_id"]])
                            after_ter=true;
                        asym_id=line_vec[_atom_site["auth_asym_id"]];
                    }
                    else if (_atom_site.count("label_asym_id"))
                    {
                        if (chain2_sele.size()) after_ter
                            =line_vec[_atom_site["label_asym_id"]]!=chain2_sele;
                        if (ter_opt>=2 && ca_idx2 && asym_id.size() && 
                            asym_id!=line_vec[_atom_site["label_asym_id"]])
                            after_ter=true;
                        asym_id=line_vec[_atom_site["label_asym_id"]];
                    }
                    if (after_ter==false || 
                        line_vec[_atom_site["group_PDB"]]=="HETATM")
                    {
                        lig_idx2++;
                        buf_all_atm_lig<<left<<setw(6)
                            <<line_vec[_atom_site["group_PDB"]]<<right
                            <<setw(5)<<(lig_idx1+lig_idx2)%100000<<' '
                            <<atom<<' '<<AA<<" B"<<resi<<inscode<<"   "
                            <<setw(8)<<line_vec[_atom_site["Cartn_x"]]
                            <<setw(8)<<line_vec[_atom_site["Cartn_y"]]
                            <<setw(8)<<line_vec[_atom_site["Cartn_z"]]
                            <<'\n';
                        if (after_ter==false &&
                            line_vec[_atom_site["group_PDB"]]=="ATOM")
                        {
                            buf_all_atm<<"ATOM  "<<setw(6)
                                <<setw(5)<<(lig_idx1+lig_idx2)%100000<<' '
                                <<atom<<' '<<AA<<" B"<<resi<<inscode<<"   "
                                <<setw(8)<<line_vec[_atom_site["Cartn_x"]]
                                <<setw(8)<<line_vec[_atom_site["Cartn_y"]]
                                <<setw(8)<<line_vec[_atom_site["Cartn_z"]]
                                <<'\n';
                            if (!mm_opt && find(resi_aln2.begin(),
                                resi_aln2.end(),resi)!=resi_aln2.end())
                            {
                                buf_atm<<"ATOM  "<<setw(6)
                                    <<setw(5)<<(lig_idx1+lig_idx2)%100000<<' '
                                    <<atom<<' '<<AA<<" B"<<resi<<inscode<<"   "
                                    <<setw(8)<<line_vec[_atom_site["Cartn_x"]]
                                    <<setw(8)<<line_vec[_atom_site["Cartn_y"]]
                                    <<setw(8)<<line_vec[_atom_site["Cartn_z"]]
                                    <<'\n';
                            }
                            if (atom==" CA " || atom==" C3'")
                            {
                                ca_idx2++;
                                buf_all<<"ATOM  "<<setw(6)
                                    <<setw(5)<<(ca_idx1+ca_idx2)%100000
                                    <<' '<<atom<<' '<<AA<<" B"<<resi<<inscode<<"   "
                                    <<setw(8)<<line_vec[_atom_site["Cartn_x"]]
                                    <<setw(8)<<line_vec[_atom_site["Cartn_y"]]
                                    <<setw(8)<<line_vec[_atom_site["Cartn_z"]]
                                    <<'\n';
                                if (!mm_opt && find(resi_aln2.begin(),
                                    resi_aln2.end(),resi)!=resi_aln2.end())
                                {
                                    buf<<"ATOM  "<<setw(6)
                                    <<setw(5)<<(ca_idx1+ca_idx2)%100000
                                    <<' '<<atom<<' '<<AA<<" B"<<resi<<inscode<<"   "
                                    <<setw(8)<<line_vec[_atom_site["Cartn_x"]]
                                    <<setw(8)<<line_vec[_atom_site["Cartn_y"]]
                                    <<setw(8)<<line_vec[_atom_site["Cartn_z"]]
                                    <<'\n';
                                    idx_vec.push_back(ca_idx1+ca_idx2);
                                }
                            }
                        }
                    }
                }

                if (fin.good()) getline(fin, line);
                else break;
            }
        }
        else if (line.size())
        {
            if (ter_opt>=1 && line.compare(0,3,"END")==0) break;
        }
    }
    fin.close();
    if (!mm_opt) buf<<"TER\n";
    buf_all<<"TER\n";
    if (!mm_opt) buf_atm<<"TER\n";
    buf_all_atm<<"TER\n";
    buf_all_atm_lig<<"TER\n";
    for (i=ca_idx1+1;i<ca_idx1+ca_idx2;i++) buf_all<<"CONECT"
        <<setw(5)<<i%100000<<setw(5)<<(i+1)%100000<<'\n';
    for (i=1;i<idx_vec.size();i++) buf<<"CONECT"
        <<setw(5)<<idx_vec[i-1]%100000<<setw(5)<<idx_vec[i]%100000<<'\n';
    idx_vec.clear();

    /* write pymol script */
    ofstream fp;
    /*
    stringstream buf_pymol;
    vector<string> pml_list;
    pml_list.push_back(fname_super+"");
    pml_list.push_back(fname_super+"_atm");
    pml_list.push_back(fname_super+"_all");
    pml_list.push_back(fname_super+"_all_atm");
    pml_list.push_back(fname_super+"_all_atm_lig");
    for (i=0;i<pml_list.size();i++)
    {
        buf_pymol<<"#!/usr/bin/env pymol\n"
            <<"load "<<pml_list[i]<<"\n"
            <<"hide all\n"
            <<((i==0 || i==2)?("show stick\n"):("show cartoon\n"))
            <<"color blue, chain A\n"
            <<"color red, chain B\n"
            <<"set ray_shadow, 0\n"
            <<"set stick_radius, 0.3\n"
            <<"set sphere_scale, 0.25\n"
            <<"show stick, not polymer\n"
            <<"show sphere, not polymer\n"
            <<"bg_color white\n"
            <<"set transparency=0.2\n"
            <<"zoom polymer\n"
            <<endl;
        fp.open((pml_list[i]+".pml").c_str());
        fp<<buf_pymol.str();
        fp.close();
        buf_pymol.str(string());
        pml_list[i].clear();
    }
    pml_list.clear();
    */
    
    /* write rasmol script */
    if (!mm_opt)
    {
        fp.open((fname_super).c_str());
        fp<<buf.str();
        fp.close();
    }
    fp.open((fname_super+"_all").c_str());
    fp<<buf_all.str();
    fp.close();
    if (!mm_opt)
    {
        fp.open((fname_super+"_atm").c_str());
        fp<<buf_atm.str();
        fp.close();
    }
    fp.open((fname_super+"_all_atm").c_str());
    fp<<buf_all_atm.str();
    fp.close();
    fp.open((fname_super+"_all_atm_lig").c_str());
    fp<<buf_all_atm_lig.str();
    fp.close();
    //fp.open((fname_super+".pdb").c_str());
    //fp<<buf_pdb.str();
    //fp.close();

    /* clear stream */
    buf.str(string());
    buf_all.str(string());
    buf_atm.str(string());
    buf_all_atm.str(string());
    buf_all_atm_lig.str(string());
    //buf_pdb.str(string());
    buf_tm.str(string());
    resi_aln1.clear();
    resi_aln2.clear();
    asym_id.clear();
    line_vec.clear();
    atom.clear();
    AA.clear();
    resi.clear();
    inscode.clear();
    model_index.clear();
}

/* extract rotation matrix based on TMscore8 */
void output_rotation_matrix(const char* fname_matrix,
    const double t[3], const double u[3][3])
{
    fstream fout;
    fout.open(fname_matrix, ios::out | ios::trunc);
    if (fout)// succeed
    {
        fout << "------ The rotation matrix to rotate Structure_1 to Structure_2 ------\n";
        char dest[1000];
        sprintf(dest, "m %18s %14s %14s %14s\n", "t[m]", "u[m][0]", "u[m][1]", "u[m][2]");
        fout << string(dest);
        for (int k = 0; k < 3; k++)
        {
            sprintf(dest, "%d %18.10f %14.10f %14.10f %14.10f\n", k, t[k], u[k][0], u[k][1], u[k][2]);
            fout << string(dest);
        }
        fout << "\nCode for rotating Structure 1 from (x,y,z) to (X,Y,Z):\n"
                "for(i=0; i<L; i++)\n"
                "{\n"
                "   X[i] = t[0] + u[0][0]*x[i] + u[0][1]*y[i] + u[0][2]*z[i];\n"
                "   Y[i] = t[1] + u[1][0]*x[i] + u[1][1]*y[i] + u[1][2]*z[i];\n"
                "   Z[i] = t[2] + u[2][0]*x[i] + u[2][1]*y[i] + u[2][2]*z[i];\n"
                "}\n";
        fout.close();
    }
    else
        cout << "Open file to output rotation matrix fail.\n";
}

//output the final results
void output_results(const string xname, const string yname,
    const string chainID1, const string chainID2,
    const int xlen, const int ylen, double t[3], double u[3][3],
    const double TM1, const double TM2,
    const double TM3, const double TM4, const double TM5,
    const double rmsd, const double d0_out, const char *seqM,
    const char *seqxA, const char *seqyA, const double Liden,
    const int n_ali8, const int L_ali, const double TM_ali,
    const double rmsd_ali, const double TM_0, const double d0_0,
    const double d0A, const double d0B, const double Lnorm_ass,
    const double d0_scale, const double d0a, const double d0u,
    const char* fname_matrix, const int outfmt_opt, const int ter_opt,
    const int mm_opt, const int split_opt, const int o_opt,
    const string fname_super, const int i_opt, const int a_opt,
    const bool u_opt, const bool d_opt, const int mirror_opt,
    const vector<string>&resi_vec1, const vector<string>&resi_vec2)
{
    if (outfmt_opt<=0)
    {
        printf("\nName of Structure_1: %s%s (to be superimposed onto Structure_2)\n",
            xname.c_str(), chainID1.c_str());
        printf("Name of Structure_2: %s%s\n", yname.c_str(), chainID2.c_str());
        printf("Length of Structure_1: %d residues\n", xlen);
        printf("Length of Structure_2: %d residues\n\n", ylen);

        if (i_opt)
            printf("User-specified initial alignment: TM/Lali/rmsd = %7.5lf, %4d, %6.3lf\n", TM_ali, L_ali, rmsd_ali);

        printf("Aligned length= %d, RMSD= %6.2f, Seq_ID=n_identical/n_aligned= %4.3f\n", n_ali8, rmsd, (n_ali8>0)?Liden/n_ali8:0);
        printf("TM-score= %6.5f (normalized by length of Structure_1: L=%d, d0=%.2f)\n", TM2, xlen, d0B);
        printf("TM-score= %6.5f (normalized by length of Structure_2: L=%d, d0=%.2f)\n", TM1, ylen, d0A);

        if (a_opt==1)
            printf("TM-score= %6.5f (if normalized by average length of two structures: L=%.1f, d0=%.2f)\n", TM3, (xlen+ylen)*0.5, d0a);
        if (u_opt)
            printf("TM-score= %6.5f (normalized by user-specified L=%.2f and d0=%.2f)\n", TM4, Lnorm_ass, d0u);
        if (d_opt)
            printf("TM-score= %6.5f (scaled by user-specified d0=%.2f, and L=%d)\n", TM5, d0_scale, ylen);
        printf("(You should use TM-score normalized by length of the reference structure)\n");
    
        //output alignment
        printf("\n(\":\" denotes residue pairs of d <%4.1f Angstrom, ", d0_out);
        printf("\".\" denotes other aligned residues)\n");
        printf("%s\n", seqxA);
        printf("%s\n", seqM);
        printf("%s\n", seqyA);
    }
    else if (outfmt_opt==1)
    {
        printf(">%s%s\tL=%d\td0=%.2f\tseqID=%.3f\tTM-score=%.5f\n",
            xname.c_str(), chainID1.c_str(), xlen, d0B, Liden/xlen, TM2);
        printf("%s\n", seqxA);
        printf(">%s%s\tL=%d\td0=%.2f\tseqID=%.3f\tTM-score=%.5f\n",
            yname.c_str(), chainID2.c_str(), ylen, d0A, Liden/ylen, TM1);
        printf("%s\n", seqyA);

        printf("# Lali=%d\tRMSD=%.2f\tseqID_ali=%.3f\n",
            n_ali8, rmsd, (n_ali8>0)?Liden/n_ali8:0);

        if (i_opt)
            printf("# User-specified initial alignment: TM=%.5lf\tLali=%4d\trmsd=%.3lf\n", TM_ali, L_ali, rmsd_ali);

        if(a_opt)
            printf("# TM-score=%.5f (normalized by average length of two structures: L=%.1f\td0=%.2f)\n", TM3, (xlen+ylen)*0.5, d0a);

        if(u_opt)
            printf("# TM-score=%.5f (normalized by user-specified L=%.2f\td0=%.2f)\n", TM4, Lnorm_ass, d0u);

        if(d_opt)
            printf("# TM-score=%.5f (scaled by user-specified d0=%.2f\tL=%d)\n", TM5, d0_scale, ylen);

        printf("$$$$\n");
    }
    else if (outfmt_opt==2)
    {
        printf("%s%s\t%s%s\t%.4f\t%.4f\t%.2f\t%4.3f\t%4.3f\t%4.3f\t%d\t%d\t%d",
            xname.c_str(), chainID1.c_str(), yname.c_str(), chainID2.c_str(),
            TM2, TM1, rmsd, Liden/xlen, Liden/ylen, (n_ali8>0)?Liden/n_ali8:0,
            xlen, ylen, n_ali8);
    }
    cout << endl;

    if (strlen(fname_matrix)) output_rotation_matrix(fname_matrix, t, u);

    if (o_opt==1)
        output_pymol(xname, yname, fname_super, t, u, ter_opt,
            mm_opt, split_opt, mirror_opt, seqM, seqxA, seqyA,
            resi_vec1, resi_vec2, chainID1, chainID2);
    else if (o_opt==2)
        output_rasmol(xname, yname, fname_super, t, u, ter_opt,
            mm_opt, split_opt, mirror_opt, seqM, seqxA, seqyA,
            resi_vec1, resi_vec2, chainID1, chainID2,
            xlen, ylen, d0A, n_ali8, rmsd, TM1, Liden);
}

void output_mTMalign_results(const string xname, const string yname,
    const string chainID1, const string chainID2,
    const int xlen, const int ylen, double t[3], double u[3][3],
    const double TM1, const double TM2,
    const double TM3, const double TM4, const double TM5,
    const double rmsd, const double d0_out, const char *seqM,
    const char *seqxA, const char *seqyA, const double Liden,
    const int n_ali8, const int L_ali, const double TM_ali,
    const double rmsd_ali, const double TM_0, const double d0_0,
    const double d0A, const double d0B, const double Lnorm_ass,
    const double d0_scale, const double d0a, const double d0u,
    const char* fname_matrix, const int outfmt_opt, const int ter_opt,
    const int mm_opt, const int split_opt, const int o_opt,
    const string fname_super, const int i_opt, const int a_opt,
    const bool u_opt, const bool d_opt, const int mirror_opt,
    const vector<string>&resi_vec1, const vector<string>&resi_vec2)
{
    if (outfmt_opt<=0)
    {
        printf("Average aligned length= %d, RMSD= %6.2f, Seq_ID=n_identical/n_aligned= %4.3f\n", n_ali8, rmsd, (n_ali8>0)?Liden/n_ali8:0);
        printf("Average TM-score= %6.5f (normalized by length of longer structure: L=%d, d0=%.2f)\n", TM2, xlen, d0B);
        printf("Average TM-score= %6.5f (normalized by length of shorter structure: L=%d, d0=%.2f)\n", TM1, ylen, d0A);

        if (a_opt==1)
            printf("Average TM-score= %6.5f (if normalized by average length of two structures: L=%.1f, d0=%.2f)\n", TM3, (xlen+ylen)*0.5, d0a);
        if (u_opt)
            printf("Average TM-score= %6.5f (normalized by average L=%.2f and d0=%.2f)\n", TM4, Lnorm_ass, d0u);
        if (d_opt)
            printf("Average TM-score= %6.5f (scaled by user-specified d0=%.2f, and L=%d)\n", TM5, d0_scale, ylen);
    
        //output alignment
        printf("In the following, seqID=n_identical/L.\n\n%s\n", seqM);
    }
    else if (outfmt_opt==1)
    {
        printf("%s\n", seqM);

        printf("# Lali=%d\tRMSD=%.2f\tseqID_ali=%.3f\n",
            n_ali8, rmsd, (n_ali8>0)?Liden/n_ali8:0);

        if (i_opt)
            printf("# User-specified initial alignment: TM=%.5lf\tLali=%4d\trmsd=%.3lf\n", TM_ali, L_ali, rmsd_ali);

        if(a_opt)
            printf("# TM-score=%.5f (normalized by average length of two structures: L=%.1f\td0=%.2f)\n", TM3, (xlen+ylen)*0.5, d0a);

        if(u_opt)
            printf("# TM-score=%.5f (normalized by average L=%.2f\td0=%.2f)\n", TM4, Lnorm_ass, d0u);

        if(d_opt)
            printf("# TM-score=%.5f (scaled by user-specified d0=%.2f\tL=%d)\n", TM5, d0_scale, ylen);

        printf("$$$$\n");
    }
    else if (outfmt_opt==2)
    {
        printf("%s%s\t%s%s\t%.4f\t%.4f\t%.2f\t%4.3f\t%4.3f\t%4.3f\t%d\t%d\t%d",
            xname.c_str(), chainID1.c_str(), yname.c_str(), chainID2.c_str(),
            TM2, TM1, rmsd, Liden/xlen, Liden/ylen, (n_ali8>0)?Liden/n_ali8:0,
            xlen, ylen, n_ali8);
    }
    cout << endl;

    if (strlen(fname_matrix)) output_rotation_matrix(fname_matrix, t, u);

    if (o_opt==1)
        output_pymol(xname, yname, fname_super, t, u, ter_opt,
            mm_opt, split_opt, mirror_opt, seqM, seqxA, seqyA,
            resi_vec1, resi_vec2, chainID1, chainID2);
    else if (o_opt==2)
        output_rasmol(xname, yname, fname_super, t, u, ter_opt,
            mm_opt, split_opt, mirror_opt, seqM, seqxA, seqyA,
            resi_vec1, resi_vec2, chainID1, chainID2,
            xlen, ylen, d0A, n_ali8, rmsd, TM1, Liden);
}

double standard_TMscore(double **r1, double **r2, double **xtm, double **ytm,
    double **xt, double **x, double **y, int xlen, int ylen, int invmap[],
    int& L_ali, double& RMSD, double D0_MIN, double Lnorm, double d0,
    double d0_search, double score_d8, double t[3], double u[3][3],
    const int mol_type)
{
    D0_MIN = 0.5;
    Lnorm = ylen;
    if (mol_type>0) // RNA
    {
        if     (Lnorm<=11) d0=0.3; 
        else if(Lnorm>11 && Lnorm<=15) d0=0.4;
        else if(Lnorm>15 && Lnorm<=19) d0=0.5;
        else if(Lnorm>19 && Lnorm<=23) d0=0.6;
        else if(Lnorm>23 && Lnorm<30)  d0=0.7;
        else d0=(0.6*pow((Lnorm*1.0-0.5), 1.0/2)-2.5);
    }
    else
    {
        if (Lnorm > 21) d0=(1.24*pow((Lnorm*1.0-15), 1.0/3) -1.8);
        else d0 = D0_MIN;
        if (d0 < D0_MIN) d0 = D0_MIN;
    }
    double d0_input = d0;// Scaled by seq_min

    double tmscore;// collected alined residues from invmap
    int n_al = 0;
    int i;
    for (int j = 0; j<ylen; j++)
    {
        i = invmap[j];
        if (i >= 0)
        {
            xtm[n_al][0] = x[i][0];
            xtm[n_al][1] = x[i][1];
            xtm[n_al][2] = x[i][2];

            ytm[n_al][0] = y[j][0];
            ytm[n_al][1] = y[j][1];
            ytm[n_al][2] = y[j][2];

            r1[n_al][0] = x[i][0];
            r1[n_al][1] = x[i][1];
            r1[n_al][2] = x[i][2];

            r2[n_al][0] = y[j][0];
            r2[n_al][1] = y[j][1];
            r2[n_al][2] = y[j][2];

            n_al++;
        }
        else if (i != -1) PrintErrorAndQuit("Wrong map!\n");
    }
    L_ali = n_al;

    Kabsch(r1, r2, n_al, 0, &RMSD, t, u);
    RMSD = sqrt( RMSD/(1.0*n_al) );
    
    int temp_simplify_step = 1;
    int temp_score_sum_method = 0;
    d0_search = d0_input;
    double rms = 0.0;
    tmscore = TMscore8_search_standard(r1, r2, xtm, ytm, xt, n_al, t, u,
        temp_simplify_step, temp_score_sum_method, &rms, d0_input,
        score_d8, d0);
    tmscore = tmscore * n_al / (1.0*Lnorm);

    return tmscore;
}

/* copy the value of t and u into t0,u0 */
void copy_t_u(double t[3], double u[3][3], double t0[3], double u0[3][3])
{
    int i,j;
    for (i=0;i<3;i++)
    {
        t0[i]=t[i];
        for (j=0;j<3;j++) u0[i][j]=u[i][j];
    }
}

/* calculate approximate TM-score given rotation matrix */
double approx_TM(const int xlen, const int ylen, const int a_opt,
    double **xa, double **ya, double t[3], double u[3][3],
    const int invmap0[], const int mol_type)
{
    double Lnorm_0=ylen; // normalized by the second protein
    if (a_opt==-2 && xlen>ylen) Lnorm_0=xlen;      // longer
    else if (a_opt==-1 && xlen<ylen) Lnorm_0=xlen; // shorter
    else if (a_opt==1) Lnorm_0=(xlen+ylen)/2.;     // average
    
    double D0_MIN;
    double Lnorm;
    double d0;
    double d0_search;
    parameter_set4final(Lnorm_0, D0_MIN, Lnorm, d0, d0_search, mol_type);
    double TMtmp=0;
    double d;
    double xtmp[3]={0,0,0};

    for(int i=0,j=0; j<ylen; j++)
    {
        i=invmap0[j];
        if(i>=0)//aligned
        {
            transform(t, u, &xa[i][0], &xtmp[0]);
            d=sqrt(dist(&xtmp[0], &ya[j][0]));
            TMtmp+=1/(1+(d/d0)*(d/d0));
            //if (d <= score_d8) TMtmp+=1/(1+(d/d0)*(d/d0));
        }
    }
    TMtmp/=Lnorm_0;
    return TMtmp;
}

void clean_up_after_approx_TM(int *invmap0, int *invmap,
    double **score, bool **path, double **val, double **xtm, double **ytm,
    double **xt, double **r1, double **r2, const int xlen, const int minlen)
{
    delete [] invmap0;
    delete [] invmap;
    DeleteArray(&score, xlen+1);
    DeleteArray(&path, xlen+1);
    DeleteArray(&val, xlen+1);
    DeleteArray(&xtm, minlen);
    DeleteArray(&ytm, minlen);
    DeleteArray(&xt, xlen);
    DeleteArray(&r1, minlen);
    DeleteArray(&r2, minlen);
    return;
}

/* Entry function for TM-align. Return TM-score calculation status:
 * 0   - full TM-score calculation 
 * 1   - terminated due to exception
 * 2-7 - pre-terminated due to low TM-score */
int TMalign_main(double **xa, double **ya,
    const char *seqx, const char *seqy, const char *secx, const char *secy,
    double t0[3], double u0[3][3],
    double &TM1, double &TM2, double &TM3, double &TM4, double &TM5,
    double &d0_0, double &TM_0,
    double &d0A, double &d0B, double &d0u, double &d0a, double &d0_out,
    string &seqM, string &seqxA, string &seqyA,
    double &rmsd0, int &L_ali, double &Liden,
    double &TM_ali, double &rmsd_ali, int &n_ali, int &n_ali8,
    const int xlen, const int ylen,
    const vector<string> sequence, const double Lnorm_ass,
    const double d0_scale, const int i_opt, const int a_opt,
    const bool u_opt, const bool d_opt, const bool fast_opt,
    const int mol_type, const double TMcut=-1)
{
    double D0_MIN;        //for d0
    double Lnorm;         //normalization length
    double score_d8,d0,d0_search,dcu0;//for TMscore search
    double t[3], u[3][3]; //Kabsch translation vector and rotation matrix
    double **score;       // Input score table for dynamic programming
    bool   **path;        // for dynamic programming  
    double **val;         // for dynamic programming  
    double **xtm, **ytm;  // for TMscore search engine
    double **xt;          //for saving the superposed version of r_1 or xtm
    double **r1, **r2;    // for Kabsch rotation

    /***********************/
    /* allocate memory     */
    /***********************/
    int minlen = min(xlen, ylen);
    NewArray(&score, xlen+1, ylen+1);
    NewArray(&path, xlen+1, ylen+1);
    NewArray(&val, xlen+1, ylen+1);
    NewArray(&xtm, minlen, 3);
    NewArray(&ytm, minlen, 3);
    NewArray(&xt, xlen, 3);
    NewArray(&r1, minlen, 3);
    NewArray(&r2, minlen, 3);

    /***********************/
    /*    parameter set    */
    /***********************/
    parameter_set4search(xlen, ylen, D0_MIN, Lnorm, 
        score_d8, d0, d0_search, dcu0);
    int simplify_step    = 40; //for simplified search engine
    int score_sum_method = 8;  //for scoring method, whether only sum over pairs with dis<score_d8

    int i;
    int *invmap0         = new int[ylen+1];
    int *invmap          = new int[ylen+1];
    double TM, TMmax=-1;
    for(i=0; i<ylen; i++) invmap0[i]=-1;

    double ddcc=0.4;
    if (Lnorm <= 40) ddcc=0.1;   //Lnorm was setted in parameter_set4search
    double local_d0_search = d0_search;

    //************************************************//
    //    get initial alignment from user's input:    //
    //    Stick to the initial alignment              //
    //************************************************//
    if (i_opt==3)// if input has set parameter for "-I"
    {
        // In the original code, this loop starts from 1, which is
        // incorrect. Fortran starts from 1 but C++ should starts from 0.
        for (int j = 0; j < ylen; j++)// Set aligned position to be "-1"
            invmap[j] = -1;

        int i1 = -1;// in C version, index starts from zero, not from one
        int i2 = -1;
        int L1 = sequence[0].size();
        int L2 = sequence[1].size();
        int L = min(L1, L2);// Get positions for aligned residues
        for (int kk1 = 0; kk1 < L; kk1++)
        {
            if (sequence[0][kk1] != '-') i1++;
            if (sequence[1][kk1] != '-')
            {
                i2++;
                if (i2 >= ylen || i1 >= xlen) kk1 = L;
                else if (sequence[0][kk1] != '-') invmap[i2] = i1;
            }
        }

        //--------------- 2. Align proteins from original alignment
        double prevD0_MIN = D0_MIN;// stored for later use
        int prevLnorm = Lnorm;
        double prevd0 = d0;
        TM_ali = standard_TMscore(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen,
            invmap, L_ali, rmsd_ali, D0_MIN, Lnorm, d0, d0_search, score_d8,
            t, u, mol_type);
        D0_MIN = prevD0_MIN;
        Lnorm = prevLnorm;
        d0 = prevd0;
        TM = detailed_search_standard(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen,
            invmap, t, u, 40, 8, local_d0_search, true, Lnorm, score_d8, d0);
        if (TM > TMmax)
        {
            TMmax = TM;
            for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
        }
    }

    /******************************************************/
    /*    get initial alignment with gapless threading    */
    /******************************************************/
    if (i_opt<=1)
    {
        get_initial(r1, r2, xtm, ytm, xa, ya, xlen, ylen, invmap0, d0,
            d0_search, fast_opt, t, u);
        TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen, invmap0,
            t, u, simplify_step, score_sum_method, local_d0_search, Lnorm,
            score_d8, d0);
        if (TM>TMmax) TMmax = TM;
        if (TMcut>0) copy_t_u(t, u, t0, u0);
        //run dynamic programing iteratively to find the best alignment
        TM = DP_iter(r1, r2, xtm, ytm, xt, path, val, xa, ya, xlen, ylen,
             t, u, invmap, 0, 2, (fast_opt)?2:30, local_d0_search,
             D0_MIN, Lnorm, d0, score_d8);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            if (TMcut>0) copy_t_u(t, u, t0, u0);
        }

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.5*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 2;
            }
        }

        /************************************************************/
        /*    get initial alignment based on secondary structure    */
        /************************************************************/
        get_initial_ss(path, val, secx, secy, xlen, ylen, invmap);
        TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen, invmap,
            t, u, simplify_step, score_sum_method, local_d0_search, Lnorm,
            score_d8, d0);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            if (TMcut>0) copy_t_u(t, u, t0, u0);
        }
        if (TM > TMmax*0.2)
        {
            TM = DP_iter(r1, r2, xtm, ytm, xt, path, val, xa, ya,
                xlen, ylen, t, u, invmap, 0, 2, (fast_opt)?2:30,
                local_d0_search, D0_MIN, Lnorm, d0, score_d8);
            if (TM>TMmax)
            {
                TMmax = TM;
                for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                if (TMcut>0) copy_t_u(t, u, t0, u0);
            }
        }

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.52*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 3;
            }
        }

        /************************************************************/
        /*    get initial alignment based on local superposition    */
        /************************************************************/
        //=initial5 in original TM-align
        if (get_initial5( r1, r2, xtm, ytm, path, val, xa, ya,
            xlen, ylen, invmap, d0, d0_search, fast_opt, D0_MIN))
        {
            TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen,
                invmap, t, u, simplify_step, score_sum_method,
                local_d0_search, Lnorm, score_d8, d0);
            if (TM>TMmax)
            {
                TMmax = TM;
                for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                if (TMcut>0) copy_t_u(t, u, t0, u0);
            }
            if (TM > TMmax*ddcc)
            {
                TM = DP_iter(r1, r2, xtm, ytm, xt, path, val, xa, ya,
                    xlen, ylen, t, u, invmap, 0, 2, 2, local_d0_search,
                    D0_MIN, Lnorm, d0, score_d8);
                if (TM>TMmax)
                {
                    TMmax = TM;
                    for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                    if (TMcut>0) copy_t_u(t, u, t0, u0);
                }
            }
        }
        else
            cerr << "\n\nWarning: initial alignment from local superposition fail!\n\n" << endl;

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.54*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 4;
            }
        }

        /********************************************************************/
        /* get initial alignment by local superposition+secondary structure */
        /********************************************************************/
        //=initial3 in original TM-align
        get_initial_ssplus(r1, r2, score, path, val, secx, secy, xa, ya,
            xlen, ylen, invmap0, invmap, D0_MIN, d0);
        TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen, invmap,
             t, u, simplify_step, score_sum_method, local_d0_search, Lnorm,
             score_d8, d0);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            if (TMcut>0) copy_t_u(t, u, t0, u0);
        }
        if (TM > TMmax*ddcc)
        {
            TM = DP_iter(r1, r2, xtm, ytm, xt, path, val, xa, ya,
                xlen, ylen, t, u, invmap, 0, 2, (fast_opt)?2:30,
                local_d0_search, D0_MIN, Lnorm, d0, score_d8);
            if (TM>TMmax)
            {
                TMmax = TM;
                for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                if (TMcut>0) copy_t_u(t, u, t0, u0);
            }
        }

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.56*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 5;
            }
        }

        /*******************************************************************/
        /*    get initial alignment based on fragment gapless threading    */
        /*******************************************************************/
        //=initial4 in original TM-align
        get_initial_fgt(r1, r2, xtm, ytm, xa, ya, xlen, ylen,
            invmap, d0, d0_search, dcu0, fast_opt, t, u);
        TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen, invmap,
            t, u, simplify_step, score_sum_method, local_d0_search, Lnorm,
            score_d8, d0);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            if (TMcut>0) copy_t_u(t, u, t0, u0);
        }
        if (TM > TMmax*ddcc)
        {
            TM = DP_iter(r1, r2, xtm, ytm, xt, path, val, xa, ya,
                xlen, ylen, t, u, invmap, 1, 2, 2, local_d0_search, D0_MIN,
                Lnorm, d0, score_d8);
            if (TM>TMmax)
            {
                TMmax = TM;
                for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                if (TMcut>0) copy_t_u(t, u, t0, u0);
            }
        }

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.58*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 6;
            }
        }
    }

    //************************************************//
    //    get initial alignment from user's input:    //
    //************************************************//
    if (i_opt>=1 && i_opt<=2)// if input has set parameter for "-i"
    {
        for (int j = 0; j < ylen; j++)// Set aligned position to be "-1"
            invmap[j] = -1;

        int i1 = -1;// in C version, index starts from zero, not from one
        int i2 = -1;
        int L1 = sequence[0].size();
        int L2 = sequence[1].size();
        int L = min(L1, L2);// Get positions for aligned residues
        for (int kk1 = 0; kk1 < L; kk1++)
        {
            if (sequence[0][kk1] != '-')
                i1++;
            if (sequence[1][kk1] != '-')
            {
                i2++;
                if (i2 >= ylen || i1 >= xlen) kk1 = L;
                else if (sequence[0][kk1] != '-') invmap[i2] = i1;
            }
        }

        //--------------- 2. Align proteins from original alignment
        double prevD0_MIN = D0_MIN;// stored for later use
        int prevLnorm = Lnorm;
        double prevd0 = d0;
        TM_ali = standard_TMscore(r1, r2, xtm, ytm, xt, xa, ya,
            xlen, ylen, invmap, L_ali, rmsd_ali, D0_MIN, Lnorm, d0,
            d0_search, score_d8, t, u, mol_type);
        D0_MIN = prevD0_MIN;
        Lnorm = prevLnorm;
        d0 = prevd0;

        TM = detailed_search_standard(r1, r2, xtm, ytm, xt, xa, ya,
            xlen, ylen, invmap, t, u, 40, 8, local_d0_search, true, Lnorm,
            score_d8, d0);
        if (TM > TMmax)
        {
            TMmax = TM;
            for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
        }
        // Different from get_initial, get_initial_ss and get_initial_ssplus
        TM = DP_iter(r1, r2, xtm, ytm, xt, path, val, xa, ya,
            xlen, ylen, t, u, invmap, 0, 2, (fast_opt)?2:30,
            local_d0_search, D0_MIN, Lnorm, d0, score_d8);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
        }
    }



    //*******************************************************************//
    //    The alignment will not be changed any more in the following    //
    //*******************************************************************//
    //check if the initial alignment is generated appropriately
    bool flag=false;
    for(i=0; i<ylen; i++)
    {
        if(invmap0[i]>=0)
        {
            flag=true;
            break;
        }
    }
    if(!flag)
    {
        cout << "There is no alignment between the two structures! "
             << "Program stop with no result!" << endl;
        TM1=TM2=TM3=TM4=TM5=0;
        return 1;
    }

    /* last TM-score pre-termination */
    if (TMcut>0)
    {
        double TMtmp=approx_TM(xlen, ylen, a_opt,
            xa, ya, t0, u0, invmap0, mol_type);

        if (TMtmp<0.6*TMcut)
        {
            TM1=TM2=TM3=TM4=TM5=TMtmp;
            clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                xtm, ytm, xt, r1, r2, xlen, minlen);
            return 7;
        }
    }

    //********************************************************************//
    //    Detailed TMscore search engine --> prepare for final TMscore    //
    //********************************************************************//
    //run detailed TMscore search engine for the best alignment, and
    //extract the best rotation matrix (t, u) for the best alignment
    simplify_step=1;
    if (fast_opt) simplify_step=40;
    score_sum_method=8;
    TM = detailed_search_standard(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen,
        invmap0, t, u, simplify_step, score_sum_method, local_d0_search,
        false, Lnorm, score_d8, d0);

    //select pairs with dis<d8 for final TMscore computation and output alignment
    int k=0;
    int *m1, *m2;
    double d;
    m1=new int[xlen]; //alignd index in x
    m2=new int[ylen]; //alignd index in y
    do_rotation(xa, xt, xlen, t, u);
    k=0;
    for(int j=0; j<ylen; j++)
    {
        i=invmap0[j];
        if(i>=0)//aligned
        {
            n_ali++;
            d=sqrt(dist(&xt[i][0], &ya[j][0]));
            if (d <= score_d8 || (i_opt == 3))
            {
                m1[k]=i;
                m2[k]=j;

                xtm[k][0]=xa[i][0];
                xtm[k][1]=xa[i][1];
                xtm[k][2]=xa[i][2];

                ytm[k][0]=ya[j][0];
                ytm[k][1]=ya[j][1];
                ytm[k][2]=ya[j][2];

                r1[k][0] = xt[i][0];
                r1[k][1] = xt[i][1];
                r1[k][2] = xt[i][2];
                r2[k][0] = ya[j][0];
                r2[k][1] = ya[j][1];
                r2[k][2] = ya[j][2];

                k++;
            }
        }
    }
    n_ali8=k;

    Kabsch(r1, r2, n_ali8, 0, &rmsd0, t, u);// rmsd0 is used for final output, only recalculate rmsd0, not t & u
    rmsd0 = sqrt(rmsd0 / n_ali8);


    //****************************************//
    //              Final TMscore             //
    //    Please set parameters for output    //
    //****************************************//
    double rmsd;
    simplify_step=1;
    score_sum_method=0;
    double Lnorm_0=ylen;


    //normalized by length of structure A
    parameter_set4final(Lnorm_0, D0_MIN, Lnorm, d0, d0_search, mol_type);
    d0A=d0;
    d0_0=d0A;
    local_d0_search = d0_search;
    TM1 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0, simplify_step,
        score_sum_method, &rmsd, local_d0_search, Lnorm, score_d8, d0);
    TM_0 = TM1;

    //normalized by length of structure B
    parameter_set4final(xlen+0.0, D0_MIN, Lnorm, d0, d0_search, mol_type);
    d0B=d0;
    local_d0_search = d0_search;
    TM2 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t, u, simplify_step,
        score_sum_method, &rmsd, local_d0_search, Lnorm, score_d8, d0);

    double Lnorm_d0;
    if (a_opt>0)
    {
        //normalized by average length of structures A, B
        Lnorm_0=(xlen+ylen)*0.5;
        parameter_set4final(Lnorm_0, D0_MIN, Lnorm, d0, d0_search, mol_type);
        d0a=d0;
        d0_0=d0a;
        local_d0_search = d0_search;

        TM3 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0,
            simplify_step, score_sum_method, &rmsd, local_d0_search, Lnorm,
            score_d8, d0);
        TM_0=TM3;
    }
    if (u_opt)
    {
        //normalized by user assigned length
        parameter_set4final(Lnorm_ass, D0_MIN, Lnorm,
            d0, d0_search, mol_type);
        d0u=d0;
        d0_0=d0u;
        Lnorm_0=Lnorm_ass;
        local_d0_search = d0_search;
        TM4 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0,
            simplify_step, score_sum_method, &rmsd, local_d0_search, Lnorm,
            score_d8, d0);
        TM_0=TM4;
    }
    if (d_opt)
    {
        //scaled by user assigned d0
        parameter_set4scale(ylen, d0_scale, Lnorm, d0, d0_search);
        d0_out=d0_scale;
        d0_0=d0_scale;
        //Lnorm_0=ylen;
        Lnorm_d0=Lnorm_0;
        local_d0_search = d0_search;
        TM5 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0,
            simplify_step, score_sum_method, &rmsd, local_d0_search, Lnorm,
            score_d8, d0);
        TM_0=TM5;
    }

    /* derive alignment from superposition */
    int ali_len=xlen+ylen; //maximum length of alignment
    seqxA.assign(ali_len,'-');
    seqM.assign( ali_len,' ');
    seqyA.assign(ali_len,'-');
    
    //do_rotation(xa, xt, xlen, t, u);
    do_rotation(xa, xt, xlen, t0, u0);

    int kk=0, i_old=0, j_old=0;
    d=0;
    Liden=0;
    for(int k=0; k<n_ali8; k++)
    {
        for(int i=i_old; i<m1[k]; i++)
        {
            //align x to gap
            seqxA[kk]=seqx[i];
            seqyA[kk]='-';
            seqM[kk]=' ';                    
            kk++;
        }

        for(int j=j_old; j<m2[k]; j++)
        {
            //align y to gap
            seqxA[kk]='-';
            seqyA[kk]=seqy[j];
            seqM[kk]=' ';
            kk++;
        }

        seqxA[kk]=seqx[m1[k]];
        seqyA[kk]=seqy[m2[k]];
        Liden+=(seqxA[kk]==seqyA[kk]);
        d=sqrt(dist(&xt[m1[k]][0], &ya[m2[k]][0]));
        if(d<d0_out) seqM[kk]=':';
        else         seqM[kk]='.';
        kk++;  
        i_old=m1[k]+1;
        j_old=m2[k]+1;
    }

    //tail
    for(int i=i_old; i<xlen; i++)
    {
        //align x to gap
        seqxA[kk]=seqx[i];
        seqyA[kk]='-';
        seqM[kk]=' ';
        kk++;
    }    
    for(int j=j_old; j<ylen; j++)
    {
        //align y to gap
        seqxA[kk]='-';
        seqyA[kk]=seqy[j];
        seqM[kk]=' ';
        kk++;
    }
    seqxA=seqxA.substr(0,kk);
    seqyA=seqyA.substr(0,kk);
    seqM =seqM.substr(0,kk);

    /* free memory */
    clean_up_after_approx_TM(invmap0, invmap, score, path, val,
        xtm, ytm, xt, r1, r2, xlen, minlen);
    delete [] m1;
    delete [] m2;
    return 0; // zero for no exception
}

/* entry function for TM-align with circular permutation
 * i_opt, a_opt, u_opt, d_opt, TMcut are not implemented yet */
int CPalign_main(double **xa, double **ya,
    const char *seqx, const char *seqy, const char *secx, const char *secy,
    double t0[3], double u0[3][3],
    double &TM1, double &TM2, double &TM3, double &TM4, double &TM5,
    double &d0_0, double &TM_0,
    double &d0A, double &d0B, double &d0u, double &d0a, double &d0_out,
    string &seqM, string &seqxA, string &seqyA,
    double &rmsd0, int &L_ali, double &Liden,
    double &TM_ali, double &rmsd_ali, int &n_ali, int &n_ali8,
    const int xlen, const int ylen,
    const vector<string> sequence, const double Lnorm_ass,
    const double d0_scale, const int i_opt, const int a_opt,
    const bool u_opt, const bool d_opt, const bool fast_opt,
    const int mol_type, const double TMcut=-1)
{
    char   *seqx_cp; // for the protein sequence 
    char   *secx_cp; // for the secondary structure 
    double **xa_cp;   // coordinates
    string seqxA_cp,seqyA_cp;  // alignment
    int    i,r;
    int    cp_point=0;    // position of circular permutation
    int    cp_aln_best=0; // amount of aligned residue in sliding window
    int    cp_aln_current;// amount of aligned residue in sliding window

    /* duplicate structure */
    NewArray(&xa_cp, xlen*2, 3);
    seqx_cp = new char[xlen*2 + 1];
    secx_cp = new char[xlen*2 + 1];
    for (r=0;r<xlen;r++)
    {
        xa_cp[r+xlen][0]=xa_cp[r][0]=xa[r][0];
        xa_cp[r+xlen][1]=xa_cp[r][1]=xa[r][1];
        xa_cp[r+xlen][2]=xa_cp[r][2]=xa[r][2];
        seqx_cp[r+xlen]=seqx_cp[r]=seqx[r];
        secx_cp[r+xlen]=secx_cp[r]=secx[r];
    }
    seqx_cp[2*xlen]=0;
    secx_cp[2*xlen]=0;
    
    /* fTM-align alignment */
    double TM1_cp,TM2_cp,TM4_cp;
    const double Lnorm_tmp=getmin(xlen,ylen);
    TMalign_main(xa_cp, ya, seqx_cp, seqy, secx_cp, secy,
        t0, u0, TM1_cp, TM2_cp, TM3, TM4_cp, TM5,
        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA_cp, seqyA_cp,
        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
        xlen*2, ylen, sequence, Lnorm_tmp, d0_scale,
        0, false, true, false, true, mol_type, -1);

    /* delete gap in seqxA_cp */
    r=0;
    seqxA=seqxA_cp;
    seqyA=seqyA_cp;
    for (i=0;i<seqxA_cp.size();i++)
    {
        if (seqxA_cp[i]!='-')
        {
            seqxA[r]=seqxA_cp[i];
            seqyA[r]=seqyA_cp[i];
            r++;
        }
    }
    seqxA=seqxA.substr(0,r);
    seqyA=seqyA.substr(0,r);

    /* count the number of aligned residues in each window
     * r - residue index in the original unaligned sequence 
     * i - position in the alignment */
    for (r=0;r<xlen-1;r++)
    {
        cp_aln_current=0;
        for (i=r;i<r+xlen;i++) cp_aln_current+=(seqyA[i]!='-');

        if (cp_aln_current>cp_aln_best)
        {
            cp_aln_best=cp_aln_current;
            cp_point=r;
        }
    }
    seqM.clear();
    seqxA.clear();
    seqyA.clear();
    seqxA_cp.clear();
    seqyA_cp.clear();
    rmsd0=Liden=n_ali=n_ali8=0;

    /* fTM-align alignment */
    TMalign_main(xa, ya, seqx, seqy, secx, secy,
        t0, u0, TM1, TM2, TM3, TM4, TM5,
        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
        xlen, ylen, sequence, Lnorm_tmp, d0_scale,
        0, false, true, false, true, mol_type, -1);

    /* do not use circular permutation of number of aligned residues is not
     * larger than sequence-order dependent alignment */
    //cout<<"cp: aln="<<cp_aln_best<<"\tTM="<<TM4_cp<<endl;
    //cout<<"TM: aln="<<n_ali8<<"\tTM="<<TM4<<endl;
    if (n_ali8>=cp_aln_best || TM4>=TM4_cp) cp_point=0;

    /* prepare structure for final alignment */
    seqM.clear();
    seqxA.clear();
    seqyA.clear();
    rmsd0=Liden=n_ali=n_ali8=0;
    if (cp_point!=0)
    {
        for (r=0;r<xlen;r++)
        {
            xa_cp[r][0]=xa_cp[r+cp_point][0];
            xa_cp[r][1]=xa_cp[r+cp_point][1];
            xa_cp[r][2]=xa_cp[r+cp_point][2];
            seqx_cp[r]=seqx_cp[r+cp_point];
            secx_cp[r]=secx_cp[r+cp_point];
        }
    }
    seqx_cp[xlen]=0;
    secx_cp[xlen]=0;

    /* test another round of alignment as concatenated alignment can
     * inflate the number of aligned residues and TM-score. e.g. 1yadA 2duaA */
    if (cp_point!=0)
    {
        TMalign_main(xa_cp, ya, seqx_cp, seqy, secx_cp, secy,
            t0, u0, TM1_cp, TM2_cp, TM3, TM4_cp, TM5,
            d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA_cp, seqyA_cp,
            rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, cp_aln_best,
            xlen, ylen, sequence, Lnorm_tmp, d0_scale,
            0, false, true, false, true, mol_type, -1);
        //cout<<"cp: aln="<<cp_aln_best<<"\tTM="<<TM4_cp<<endl;
        if (n_ali8>=cp_aln_best || TM4>=TM4_cp)
        {
            cp_point=0;
            for (r=0;r<xlen;r++)
            {
                xa_cp[r][0]=xa[r][0];
                xa_cp[r][1]=xa[r][1];
                xa_cp[r][2]=xa[r][2];
                seqx_cp[r]=seqx[r];
                secx_cp[r]=secx[r];
            }
        }
    }

    /* full TM-align */
    TMalign_main(xa_cp, ya, seqx_cp, seqy, secx_cp, secy,
        t0, u0, TM1, TM2, TM3, TM4, TM5,
        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA_cp, seqyA_cp,
        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
        xlen, ylen, sequence, Lnorm_ass, d0_scale,
        i_opt, a_opt, u_opt, d_opt, fast_opt, mol_type, TMcut);

    /* correct alignment
     * r - residue index in the original unaligned sequence 
     * i - position in the alignment */
    if (cp_point>0)
    {
        r=0;
        for (i=0;i<seqxA_cp.size();i++)
        {
            r+=(seqxA_cp[i]!='-');
            if (r>=(xlen-cp_point)) 
            {
                i++;
                break;
            }
        }
        seqxA=seqxA_cp.substr(0,i)+'*'+seqxA_cp.substr(i);
        seqM =seqM.substr(0,i)    +' '+seqM.substr(i);
        seqyA=seqyA_cp.substr(0,i)+'-'+seqyA_cp.substr(i);
    }
    else
    {
        seqxA=seqxA_cp;
        seqyA=seqyA_cp;
    }

    /* clean up */
    delete[]seqx_cp;
    delete[]secx_cp;
    DeleteArray(&xa_cp,xlen*2);
    seqxA_cp.clear();
    seqyA_cp.clear();
    return cp_point;
}

bool output_cp(const string&xname, const string&yname,
    const string &seqxA, const string &seqyA, const int outfmt_opt)
{
    int r;
    bool after_cp=false;
    int left_num=0;
    int right_num=0;
    int left_aln_num=0;
    int right_aln_num=0;
    for (r=0;r<seqxA.size();r++)
    {
        if      (seqxA[r]=='-') continue;
        else if (seqxA[r]=='*') after_cp=true;
        else 
        {
            if (after_cp)
            {
                right_num++;
                right_aln_num+=(seqyA[r]!='-');
            }
            else
            {
                left_num++;
                left_aln_num+=(seqyA[r]!='-');
            }
        }
    }
    if (after_cp==false)
    {
        if (outfmt_opt<=0) cout<<"No CP"<<endl;
        else if (outfmt_opt==1) cout<<"#No CP"<<endl;
        else if (outfmt_opt==2) cout<<"@"<<xname<<'\t'<<yname<<'\t'<<"No CP"<<endl;
    }
    else
    {
        if (outfmt_opt<=0) cout<<"CP point in structure_1: "<<left_num<<'/'<<right_num<<'\n'
            <<"CP point in structure_1 alignment: "<<left_aln_num<<'/'<<right_aln_num<<endl;
        else if (outfmt_opt==1) cout<<"#CP_in_seq="<<left_num<<'/'<<right_num
                                    <<"\tCP_in_aln="<<left_aln_num<<'/'<<right_aln_num<<endl;
        else if (outfmt_opt==2) cout<<"@"<<xname<<'\t'<<yname<<'\t'<<left_num
            <<'/'<<right_num<<'\t'<<left_aln_num<<'/'<<right_aln_num<<endl;

    }
    return after_cp;
}

/* se.h */
/* entry function for se
 * outfmt_opt>=2 should not parse sequence alignment 
 * u_opt corresponds to option -L
 *       if u_opt==2, use d0 from Lnorm_ass for alignment
 * */
int se_main(
    double **xa, double **ya, const char *seqx, const char *seqy,
    double &TM1, double &TM2, double &TM3, double &TM4, double &TM5,
    double &d0_0, double &TM_0,
    double &d0A, double &d0B, double &d0u, double &d0a, double &d0_out,
    string &seqM, string &seqxA, string &seqyA,
    double &rmsd0, int &L_ali, double &Liden,
    double &TM_ali, double &rmsd_ali, int &n_ali, int &n_ali8,
    const int xlen, const int ylen, const vector<string> &sequence,
    const double Lnorm_ass, const double d0_scale, const bool i_opt,
    const bool a_opt, const int u_opt, const bool d_opt, const int mol_type,
    const int outfmt_opt, int *invmap)
{
    double D0_MIN;        //for d0
    double Lnorm;         //normalization length
    double score_d8,d0,d0_search,dcu0;//for TMscore search
    double **score;       // Input score table for dynamic programming
    bool   **path;        // for dynamic programming  
    double **val;         // for dynamic programming  

    int *m1=NULL;
    int *m2=NULL;
    double d;
    if (outfmt_opt<2)
    {
        m1=new int[xlen]; //alignd index in x
        m2=new int[ylen]; //alignd index in y
    }

    /***********************/
    /* allocate memory     */
    /***********************/
    NewArray(&score, xlen+1, ylen+1);
    NewArray(&path, xlen+1, ylen+1);
    NewArray(&val, xlen+1, ylen+1);
    //int *invmap          = new int[ylen+1];

    /* set d0 */
    parameter_set4search(xlen, ylen, D0_MIN, Lnorm,
        score_d8, d0, d0_search, dcu0); // set score_d8
    parameter_set4final(xlen, D0_MIN, Lnorm,
        d0B, d0_search, mol_type); // set d0B
    parameter_set4final(ylen, D0_MIN, Lnorm,
        d0A, d0_search, mol_type); // set d0A
    if (a_opt)
        parameter_set4final((xlen+ylen)*0.5, D0_MIN, Lnorm,
            d0a, d0_search, mol_type); // set d0a
    if (u_opt)
    {
        parameter_set4final(Lnorm_ass, D0_MIN, Lnorm,
            d0u, d0_search, mol_type); // set d0u
        if (u_opt==2)
        {
            parameter_set4search(Lnorm_ass, Lnorm_ass, D0_MIN, Lnorm,
                score_d8, d0, d0_search, dcu0); // set score_d8
        }
    }

    /* perform alignment */
    for(int j=0; j<ylen; j++) invmap[j]=-1;
    if (!i_opt) NWDP_SE(path, val, xa, ya, xlen, ylen, d0*d0, 0, invmap);
    else
    {
        int i1 = -1;// in C version, index starts from zero, not from one
        int i2 = -1;
        int L1 = sequence[0].size();
        int L2 = sequence[1].size();
        int L = min(L1, L2);// Get positions for aligned residues
        for (int kk1 = 0; kk1 < L; kk1++)
        {
            if (sequence[0][kk1] != '-') i1++;
            if (sequence[1][kk1] != '-')
            {
                i2++;
                if (i2 >= ylen || i1 >= xlen) kk1 = L;
                else if (sequence[0][kk1] != '-') invmap[i2] = i1;
            }
        }
    }

    rmsd0=TM1=TM2=TM3=TM4=TM5=0;
    int k=0;
    n_ali=0;
    n_ali8=0;
    for(int i=0,j=0; j<ylen; j++)
    {
        i=invmap[j];
        if(i>=0)//aligned
        {
            n_ali++;
            d=sqrt(dist(&xa[i][0], &ya[j][0]));
            if (d <= score_d8 || i_opt)
            {
                if (outfmt_opt<2)
                {
                    m1[k]=i;
                    m2[k]=j;
                }
                k++;
                TM2+=1/(1+(d/d0B)*(d/d0B)); // chain_1
                TM1+=1/(1+(d/d0A)*(d/d0A)); // chain_2
                if (a_opt) TM3+=1/(1+(d/d0a)*(d/d0a)); // -a
                if (u_opt) TM4+=1/(1+(d/d0u)*(d/d0u)); // -u
                if (d_opt) TM5+=1/(1+(d/d0_scale)*(d/d0_scale)); // -d
                rmsd0+=d*d;
            }
        }
    }
    n_ali8=k;
    TM2/=xlen;
    TM1/=ylen;
    TM3/=(xlen+ylen)*0.5;
    TM4/=Lnorm_ass;
    TM5/=ylen;
    if (n_ali8) rmsd0=sqrt(rmsd0/n_ali8);

    if (outfmt_opt>=2)
    {
        DeleteArray(&score, xlen+1);
        DeleteArray(&path, xlen+1);
        DeleteArray(&val, xlen+1);
        return 0;
    }

    /* extract aligned sequence */
    int ali_len=xlen+ylen; //maximum length of alignment
    seqxA.assign(ali_len,'-');
    seqM.assign( ali_len,' ');
    seqyA.assign(ali_len,'-');
    
    int kk=0, i_old=0, j_old=0;
    d=0;
    Liden=0;
    for(int k=0; k<n_ali8; k++)
    {
        for(int i=i_old; i<m1[k]; i++)
        {
            //align x to gap
            seqxA[kk]=seqx[i];
            seqyA[kk]='-';
            seqM[kk]=' ';                    
            kk++;
        }

        for(int j=j_old; j<m2[k]; j++)
        {
            //align y to gap
            seqxA[kk]='-';
            seqyA[kk]=seqy[j];
            seqM[kk]=' ';
            kk++;
        }

        seqxA[kk]=seqx[m1[k]];
        seqyA[kk]=seqy[m2[k]];
        Liden+=(seqxA[kk]==seqyA[kk]);
        d=sqrt(dist(&xa[m1[k]][0], &ya[m2[k]][0]));
        if(d<d0_out) seqM[kk]=':';
        else         seqM[kk]='.';
        kk++;  
        i_old=m1[k]+1;
        j_old=m2[k]+1;
    }

    //tail
    for(int i=i_old; i<xlen; i++)
    {
        //align x to gap
        seqxA[kk]=seqx[i];
        seqyA[kk]='-';
        seqM[kk]=' ';
        kk++;
    }    
    for(int j=j_old; j<ylen; j++)
    {
        //align y to gap
        seqxA[kk]='-';
        seqyA[kk]=seqy[j];
        seqM[kk]=' ';
        kk++;
    }
    seqxA=seqxA.substr(0,kk);
    seqyA=seqyA.substr(0,kk);
    seqM =seqM.substr(0,kk);

    /* free memory */
    //delete [] invmap;
    delete [] m1;
    delete [] m2;
    DeleteArray(&score, xlen+1);
    DeleteArray(&path, xlen+1);
    DeleteArray(&val, xlen+1);
    return 0; // zero for no exception
}

/* MMalign.h */
/* count the number of nucleic acid chains (na_chain_num) and
 * protein chains (aa_chain_num) in a complex */
int count_na_aa_chain_num(int &na_chain_num,int &aa_chain_num,
    const vector<int>&mol_vec)
{
    na_chain_num=0;
    aa_chain_num=0;
    for (size_t i=0;i<mol_vec.size();i++)
    {
        if (mol_vec[i]>0) na_chain_num++;
        else              aa_chain_num++;
    }
    return na_chain_num+aa_chain_num;
}

/* adjust chain assignment for dimer-dimer alignment 
 * return true if assignment is adjusted */
bool adjust_dimer_assignment(        
    const vector<vector<vector<double> > >&xa_vec,
    const vector<vector<vector<double> > >&ya_vec,
    const vector<int>&xlen_vec, const vector<int>&ylen_vec,
    const vector<int>&mol_vec1, const vector<int>&mol_vec2,
    int *assign1_list, int *assign2_list,
    const vector<vector<string> >&seqxA_mat,
    const vector<vector<string> >&seqyA_mat)
{
    /* check currently assigned chains */
    int i1,i2,j1,j2;
    i1=i2=j1=j2=-1;    
    int chain1_num=xa_vec.size();
    int i,j;
    for (i=0;i<chain1_num;i++)
    {
        if (assign1_list[i]>=0)
        {
            if (i1<0)
            {
                i1=i;
                j1=assign1_list[i1];
            }
            else
            {
                i2=i;
                j2=assign1_list[i2];
            }
        }
    }

    /* normalize d0 by L */
    int xlen=xlen_vec[i1]+xlen_vec[i2];
    int ylen=ylen_vec[j1]+ylen_vec[j2];
    int mol_type=mol_vec1[i1]+mol_vec1[i2]+
                 mol_vec2[j1]+mol_vec2[j2];
    double D0_MIN, d0, d0_search;
    double Lnorm=getmin(xlen,ylen);
    parameter_set4final(getmin(xlen,ylen), D0_MIN, Lnorm, d0, 
        d0_search, mol_type);

    double **xa,**ya, **xt;
    NewArray(&xa, xlen, 3);
    NewArray(&ya, ylen, 3);
    NewArray(&xt, xlen, 3);

    double RMSD = 0;
    double dd   = 0;
    double t[3];
    double u[3][3];
    size_t L_ali=0; // index of residue in aligned region
    size_t r=0;     // index of residue in full alignment

    /* total score using current assignment */
    L_ali=0;
    i=j=-1;
    for (r=0;r<seqxA_mat[i1][j1].size();r++)
    {
        i+=(seqxA_mat[i1][j1][r]!='-');
        j+=(seqyA_mat[i1][j1][r]!='-');
        if (seqxA_mat[i1][j1][r]=='-' || seqyA_mat[i1][j1][r]=='-') continue;
        xa[L_ali][0]=xa_vec[i1][i][0];
        xa[L_ali][1]=xa_vec[i1][i][1];
        xa[L_ali][2]=xa_vec[i1][i][2];
        ya[L_ali][0]=ya_vec[j1][j][0];
        ya[L_ali][1]=ya_vec[j1][j][1];
        ya[L_ali][2]=ya_vec[j1][j][2];
        L_ali++;
    }
    i=j=-1;
    for (r=0;r<seqxA_mat[i2][j2].size();r++)
    {
        i+=(seqxA_mat[i2][j2][r]!='-');
        j+=(seqyA_mat[i2][j2][r]!='-');
        if (seqxA_mat[i2][j2][r]=='-' || seqyA_mat[i2][j2][r]=='-') continue;
        xa[L_ali][0]=xa_vec[i2][i][0];
        xa[L_ali][1]=xa_vec[i2][i][1];
        xa[L_ali][2]=xa_vec[i2][i][2];
        ya[L_ali][0]=ya_vec[j2][j][0];
        ya[L_ali][1]=ya_vec[j2][j][1];
        ya[L_ali][2]=ya_vec[j2][j][2];
        L_ali++;
    }

    Kabsch(xa, ya, L_ali, 1, &RMSD, t, u);
    do_rotation(xa, xt, L_ali, t, u);

    double total_score1=0;
    for (r=0;r<L_ali;r++)
    {
        dd=dist(xt[r],ya[r]);
        total_score1+=1/(1+dd/d0*d0);
    }
    total_score1/=Lnorm;

    /* total score using reversed assignment */
    L_ali=0;
    i=j=-1;
    for (r=0;r<seqxA_mat[i1][j2].size();r++)
    {
        i+=(seqxA_mat[i1][j2][r]!='-');
        j+=(seqyA_mat[i1][j2][r]!='-');
        if (seqxA_mat[i1][j2][r]=='-' || seqyA_mat[i1][j2][r]=='-') continue;
        xa[L_ali][0]=xa_vec[i1][i][0];
        xa[L_ali][1]=xa_vec[i1][i][1];
        xa[L_ali][2]=xa_vec[i1][i][2];
        ya[L_ali][0]=ya_vec[j2][j][0];
        ya[L_ali][1]=ya_vec[j2][j][1];
        ya[L_ali][2]=ya_vec[j2][j][2];
        L_ali++;
    }
    i=j=-1;
    for (r=0;r<seqxA_mat[i2][j1].size();r++)
    {
        i+=(seqxA_mat[i2][j1][r]!='-');
        j+=(seqyA_mat[i2][j1][r]!='-');
        if (seqxA_mat[i2][j1][r]=='-' || seqyA_mat[i2][j1][r]=='-') continue;
        xa[L_ali][0]=xa_vec[i2][i][0];
        xa[L_ali][1]=xa_vec[i2][i][1];
        xa[L_ali][2]=xa_vec[i2][i][2];
        ya[L_ali][0]=ya_vec[j1][j][0];
        ya[L_ali][1]=ya_vec[j1][j][1];
        ya[L_ali][2]=ya_vec[j1][j][2];
        L_ali++;
    }

    Kabsch(xa, ya, L_ali, 1, &RMSD, t, u);
    do_rotation(xa, xt, L_ali, t, u);

    double total_score2=0;
    for (r=0;r<L_ali;r++)
    {
        dd=dist(xt[r],ya[r]);
        total_score2+=1/(1+dd/d0*d0);
    }
    total_score2/=Lnorm;

    /* swap chain assignment */
    if (total_score1<total_score2)
    {
        assign1_list[i1]=j2;
        assign1_list[i2]=j1;
        assign2_list[j1]=i2;
        assign2_list[j2]=i1;
    }

    /* clean up */
    DeleteArray(&xa, xlen);
    DeleteArray(&ya, ylen);
    DeleteArray(&xt, xlen);
    return total_score1<total_score2;
}

/* count how many chains are paired */
int count_assign_pair(int *assign1_list,const int chain1_num)
{
    int pair_num=0;
    int i;
    for (i=0;i<chain1_num;i++) pair_num+=(assign1_list[i]>=0);
    return pair_num;
}


/* assign chain-chain correspondence */
double enhanced_greedy_search(double **TMave_mat,int *assign1_list,
    int *assign2_list, const int chain1_num, const int chain2_num)
{
    double total_score=0;
    double tmp_score=0;
    int i,j;
    int maxi=0;
    int maxj=0;

    /* initialize parameters */
    for (i=0;i<chain1_num;i++) assign1_list[i]=-1;
    for (j=0;j<chain2_num;j++) assign2_list[j]=-1;

    /* greedy assignment: in each iteration, the highest chain pair is
     * assigned, until no assignable chain is left */
    while(1)
    {
        tmp_score=-1;
        for (i=0;i<chain1_num;i++)
        {
            if (assign1_list[i]>=0) continue;
            for (j=0;j<chain2_num;j++)
            {
                if (assign2_list[j]>=0 || TMave_mat[i][j]<=0) continue;
                if (TMave_mat[i][j]>tmp_score) 
                {
                    maxi=i;
                    maxj=j;
                    tmp_score=TMave_mat[i][j];
                }
            }
        }
        if (tmp_score<=0) break; // error: no assignable chain
        assign1_list[maxi]=maxj;
        assign2_list[maxj]=maxi;
        total_score+=tmp_score;
    }
    if (total_score<=0) return total_score; // error: no assignable chain
    //cout<<"assign1_list={";
    //for (i=0;i<chain1_num;i++) cout<<assign1_list[i]<<","; cout<<"}"<<endl;
    //cout<<"assign2_list={";
    //for (j=0;j<chain2_num;j++) cout<<assign2_list[j]<<","; cout<<"}"<<endl;

    /* iterative refinemnt */
    double delta_score;
    int *assign1_tmp=new int [chain1_num];
    int *assign2_tmp=new int [chain2_num];
    for (i=0;i<chain1_num;i++) assign1_tmp[i]=assign1_list[i];
    for (j=0;j<chain2_num;j++) assign2_tmp[j]=assign2_list[j];
    int old_i=-1;
    int old_j=-1;

    for (int iter=0;iter<getmin(chain1_num,chain2_num)*5;iter++)
    {
        delta_score=-1;
        for (i=0;i<chain1_num;i++)
        {
            old_j=assign1_list[i];
            for (j=0;j<chain2_num;j++)
            {
                // attempt to swap (i,old_j=assign1_list[i]) with (i,j)
                if (j==assign1_list[i] || TMave_mat[i][j]<=0) continue;
                old_i=assign2_list[j];

                assign1_tmp[i]=j;
                if (old_i>=0) assign1_tmp[old_i]=old_j;
                assign2_tmp[j]=i;
                if (old_j>=0) assign2_tmp[old_j]=old_i;

                delta_score=TMave_mat[i][j];
                if (old_j>=0) delta_score-=TMave_mat[i][old_j];
                if (old_i>=0) delta_score-=TMave_mat[old_i][j];
                if (old_i>=0 && old_j>=0) delta_score+=TMave_mat[old_i][old_j];

                if (delta_score>0) // successful swap
                {
                    assign1_list[i]=j;
                    if (old_i>=0) assign1_list[old_i]=old_j;
                    assign2_list[j]=i;
                    if (old_j>=0) assign2_list[old_j]=old_i;
                    total_score+=delta_score;
                    break;
                }
                else
                {
                    assign1_tmp[i]=assign1_list[i];
                    if (old_i>=0) assign1_tmp[old_i]=assign1_list[old_i];
                    assign2_tmp[j]=assign2_list[j];
                    if (old_j>=0) assign2_tmp[old_j]=assign2_list[old_j];
                }
            }
            if (delta_score>0) break;
        }
        if (delta_score<=0) break; // cannot swap any chain pair
    }

    /* clean up */
    delete[]assign1_tmp;
    delete[]assign2_tmp;
    return total_score;
}

double calculate_centroids(const vector<vector<vector<double> > >&a_vec,
    const int chain_num, double ** centroids)
{
    int L=0;
    int c,r; // index of chain and residue
    for (c=0; c<chain_num; c++)
    {
        centroids[c][0]=0;
        centroids[c][1]=0;
        centroids[c][2]=0;
        L=a_vec[c].size();
        for (r=0; r<L; r++)
        {
            centroids[c][0]+=a_vec[c][r][0];
            centroids[c][1]+=a_vec[c][r][1];
            centroids[c][2]+=a_vec[c][r][2];
        }
        centroids[c][0]/=L;
        centroids[c][1]/=L;
        centroids[c][2]/=L;
        //cout<<centroids[c][0]<<'\t'
            //<<centroids[c][1]<<'\t'
            //<<centroids[c][2]<<endl;
    }

    vector<double> d0_vec(chain_num,-1);
    int c2=0;
    double d0MM=0;
    for (c=0; c<chain_num; c++)
    {
        for (c2=0; c2<chain_num; c2++)
        {
            if (c2==c) continue;
            d0MM=sqrt(dist(centroids[c],centroids[c2]));
            if (d0_vec[c]<=0) d0_vec[c]=d0MM;
            else d0_vec[c]=getmin(d0_vec[c], d0MM);
        }
    }
    d0MM=0;
    for (c=0; c<chain_num; c++) d0MM+=d0_vec[c];
    d0MM/=chain_num;
    d0_vec.clear();
    //cout<<d0MM<<endl;
    return d0MM;
}

/* calculate MMscore of aligned chains
 * MMscore = sum(TMave_mat[i][j]) * sum(1/(1+dij^2/d0MM^2)) 
 *         / (L* getmin(chain1_num,chain2_num))
 * dij is the centroid distance between chain pair i and j
 * d0MM is scaling factor. TMave_mat[i][j] is the TM-score between
 * chain pair i and j multiple by getmin(Li*Lj) */
double calMMscore(double **TMave_mat,int *assign1_list,
    const int chain1_num, const int chain2_num, double **xcentroids,
    double **ycentroids, const double d0MM, double **r1, double **r2,
    double **xt, double t[3], double u[3][3], const int L)
{
    int Nali=0; // number of aligned chain
    int i,j;
    double MMscore=0;
    for (i=0;i<chain1_num;i++)
    {
        j=assign1_list[i];
        if (j<0) continue;

        r1[Nali][0]=xcentroids[i][0];
        r1[Nali][1]=xcentroids[i][1];
        r1[Nali][2]=xcentroids[i][2];

        r2[Nali][0]=ycentroids[j][0];
        r2[Nali][1]=ycentroids[j][1];
        r2[Nali][2]=ycentroids[j][2];

        Nali++;
        MMscore+=TMave_mat[i][j];
    }
    MMscore/=L;

    double RMSD = 0;
    double TMscore=0;
    if (Nali>=3)
    {
        /* Kabsch superposition */
        Kabsch(r1, r2, Nali, 1, &RMSD, t, u);
        do_rotation(r1, xt, Nali, t, u);

        /* calculate pseudo-TMscore */
        double dd=0;
        for (i=0;i<Nali;i++)
        {
            dd=dist(xt[i], r2[i]);
            TMscore+=1/(1+dd/(d0MM*d0MM));
        }
    }
    else if (Nali==2)
    {
        double dd=dist(r1[0],r2[0]);
        TMscore=1/(1+dd/(d0MM*d0MM));
    }
    else TMscore=1; // only one aligned chain.
    TMscore/=getmin(chain1_num,chain2_num);
    MMscore*=TMscore;
    return MMscore;
}

/* reassign chain-chain correspondence, specific for homooligomer */
double homo_refined_greedy_search(double **TMave_mat,int *assign1_list,
    int *assign2_list, const int chain1_num, const int chain2_num,
    double **xcentroids, double **ycentroids, const double d0MM,
    const int L, double **ut_mat)
{
    double MMscore_max=0;
    double MMscore=0;
    int i,j;
    int c1,c2;
    int max_i=-1; // the chain pair whose monomer u t yields highest MMscore
    int max_j=-1;

    int chain_num=getmin(chain1_num,chain2_num);
    int *assign1_tmp=new int [chain1_num];
    int *assign2_tmp=new int [chain2_num];
    double **xt;
    NewArray(&xt, chain1_num, 3);
    double t[3];
    double u[3][3];
    int ui,uj,ut_idx;
    double TMscore=0; // pseudo TM-score
    double TMsum  =0;
    double TMnow  =0;
    double TMmax  =0;
    double dd=0;

    size_t  total_pair=chain1_num*chain2_num; // total pair
    double *ut_tmc_mat=new double [total_pair]; // chain level TM-score
    vector<pair<double,int> > ut_tm_vec(total_pair,make_pair(0.0,0)); // product of both

    for (c1=0;c1<chain1_num;c1++)
    {
        for (c2=0;c2<chain2_num;c2++)
        {
            if (TMave_mat[c1][c2]<=0) continue;
            ut_idx=c1*chain2_num+c2;
            for (ui=0;ui<3;ui++)
                for (uj=0;uj<3;uj++) u[ui][uj]=ut_mat[ut_idx][ui*3+uj];
            for (uj=0;uj<3;uj++) t[uj]=ut_mat[ut_idx][9+uj];
            
            do_rotation(xcentroids, xt, chain1_num, t, u);

            for (i=0;i<chain1_num;i++) assign1_tmp[i]=-1;
            for (j=0;j<chain2_num;j++) assign2_tmp[j]=-1;


            for (i=0;i<chain1_num;i++)
            {
                for (j=0;j<chain2_num;j++)
                {
                    ut_idx=i*chain2_num+j;
                    ut_tmc_mat[ut_idx]=0;
                    ut_tm_vec[ut_idx].first=-1;
                    ut_tm_vec[ut_idx].second=ut_idx;
                    if (TMave_mat[i][j]<=0) continue;
                    dd=dist(xt[i],ycentroids[j]);
                    ut_tmc_mat[ut_idx]=1/(1+dd/(d0MM*d0MM));
                    ut_tm_vec[ut_idx].first=
                        ut_tmc_mat[ut_idx]*TMave_mat[i][j];
                    //cout<<"TM["<<ut_idx<<"]="<<ut_tm_vec[ut_idx].first<<endl;
                }
            }
            //cout<<"sorting "<<total_pair<<" chain pairs"<<endl;

            /* initial assignment */
            assign1_tmp[c1]=c2;
            assign2_tmp[c2]=c1;
            TMsum=TMave_mat[c1][c2];
            TMscore=ut_tmc_mat[c1*chain2_num+c2];

            /* further assignment */
            sort(ut_tm_vec.begin(), ut_tm_vec.end()); // sort in ascending order
            for (ut_idx=total_pair-1;ut_idx>=0;ut_idx--)
            {
                j=ut_tm_vec[ut_idx].second % chain2_num;
                i=int(ut_tm_vec[ut_idx].second / chain2_num);
                if (TMave_mat[i][j]<=0) break;
                if (assign1_tmp[i]>=0 || assign2_tmp[j]>=0) continue;
                assign1_tmp[i]=j;
                assign2_tmp[j]=i;
                TMsum+=TMave_mat[i][j];
                TMscore+=ut_tmc_mat[i*chain2_num+j];
                //cout<<"ut_idx="<<ut_tm_vec[ut_idx].second
                    //<<"\ti="<<i<<"\tj="<<j<<"\ttm="<<ut_tm_vec[ut_idx].first<<endl;
            }

            /* final MMscore */
            MMscore=(TMsum/L)*(TMscore/chain_num);
            if (max_i<0 || max_j<0 || MMscore>MMscore_max)
            {
                max_i=c1;
                max_j=c2;
                MMscore_max=MMscore;
                for (i=0;i<chain1_num;i++) assign1_list[i]=assign1_tmp[i];
                for (j=0;j<chain2_num;j++) assign2_list[j]=assign2_tmp[j];
                //cout<<"TMsum/L="<<TMsum/L<<endl;
                //cout<<"TMscore/chain_num="<<TMscore/chain_num<<endl;
                //cout<<"MMscore="<<MMscore<<endl;
                //cout<<"assign1_list={";
                //for (i=0;i<chain1_num;i++) 
                    //cout<<assign1_list[i]<<","; cout<<"}"<<endl;
                //cout<<"assign2_list={";
                //for (j=0;j<chain2_num;j++)
                    //cout<<assign2_list[j]<<","; cout<<"}"<<endl;
            }
        }
    }

    /* clean up */
    delete[]assign1_tmp;
    delete[]assign2_tmp;
    delete[]ut_tmc_mat;
    ut_tm_vec.clear();
    DeleteArray(&xt, chain1_num);
    return MMscore;
}

/* reassign chain-chain correspondence, specific for heterooligomer */
double hetero_refined_greedy_search(double **TMave_mat,int *assign1_list,
    int *assign2_list, const int chain1_num, const int chain2_num,
    double **xcentroids, double **ycentroids, const double d0MM, const int L)
{
    double MMscore_old=0;
    double MMscore=0;
    int i,j;

    double **r1;
    double **r2;
    double **xt;
    int chain_num=getmin(chain1_num,chain2_num);
    NewArray(&r1, chain_num, 3);
    NewArray(&r2, chain_num, 3);
    NewArray(&xt, chain_num, 3);
    double t[3];
    double u[3][3];

    /* calculate MMscore */
    MMscore=MMscore_old=calMMscore(TMave_mat, assign1_list, chain1_num,
        chain2_num, xcentroids, ycentroids, d0MM, r1, r2, xt, t, u, L);
    //cout<<"MMscore="<<MMscore<<endl;
    //cout<<"TMave_mat="<<endl;
    //for (i=0;i<chain1_num;i++)
    //{
        //for (j=0; j<chain2_num; j++)
        //{
            //if (j<chain2_num-1) cout<<TMave_mat[i][j]<<'\t';
            //else                cout<<TMave_mat[i][j]<<endl;
        //}
    //}

    /* iteratively refine chain assignment. in each iteration, attempt
     * to swap (i,old_j=assign1_list[i]) with (i,j) */
    double delta_score=-1;
    int *assign1_tmp=new int [chain1_num];
    int *assign2_tmp=new int [chain2_num];
    for (i=0;i<chain1_num;i++) assign1_tmp[i]=assign1_list[i];
    for (j=0;j<chain2_num;j++) assign2_tmp[j]=assign2_list[j];
    int old_i=-1;
    int old_j=-1;

    //cout<<"assign1_list={";
    //for (i=0;i<chain1_num;i++) cout<<assign1_list[i]<<","; cout<<"}"<<endl;
    //cout<<"assign2_list={";
    //for (j=0;j<chain2_num;j++) cout<<assign2_list[j]<<","; cout<<"}"<<endl;

    for (int iter=0;iter<chain1_num*chain2_num;iter++)
    {
        delta_score=-1;
        for (i=0;i<chain1_num;i++)
        {
            old_j=assign1_list[i];
            for (j=0;j<chain2_num;j++)
            {
                if (j==assign1_list[i] || TMave_mat[i][j]<=0) continue;
                old_i=assign2_list[j];

                assign1_tmp[i]=j;
                if (old_i>=0) assign1_tmp[old_i]=old_j;
                assign2_tmp[j]=i;
                if (old_j>=0) assign2_tmp[old_j]=old_i;
                
                MMscore=calMMscore(TMave_mat, assign1_tmp, chain1_num,
                    chain2_num, xcentroids, ycentroids, d0MM,
                    r1, r2, xt, t, u, L);

                //cout<<"(i,j,old_i,old_j,MMscore)=("<<i<<","<<j<<","
                    //<<old_i<<","<<old_j<<","<<MMscore<<")"<<endl;

                if (MMscore>MMscore_old) // successful swap
                {
                    assign1_list[i]=j;
                    if (old_i>=0) assign1_list[old_i]=old_j;
                    assign2_list[j]=i;
                    if (old_j>=0) assign2_list[old_j]=old_i;
                    delta_score=(MMscore-MMscore_old);
                    MMscore_old=MMscore;
                    //cout<<"MMscore="<<MMscore<<endl;
                    break;
                }
                else
                {
                    assign1_tmp[i]=assign1_list[i];
                    if (old_i>=0) assign1_tmp[old_i]=assign1_list[old_i];
                    assign2_tmp[j]=assign2_list[j];
                    if (old_j>=0) assign2_tmp[old_j]=assign2_list[old_j];
                }
            }
        }
        //cout<<"iter="<<iter<<endl;
        //cout<<"assign1_list={";
        //for (i=0;i<chain1_num;i++) cout<<assign1_list[i]<<","; cout<<"}"<<endl;
        //cout<<"assign2_list={";
        //for (j=0;j<chain2_num;j++) cout<<assign2_list[j]<<","; cout<<"}"<<endl;
        if (delta_score<=0) break; // cannot swap any chain pair
    }
    MMscore=MMscore_old;
    //cout<<"MMscore="<<MMscore<<endl;

    /* clean up */
    delete[]assign1_tmp;
    delete[]assign2_tmp;
    DeleteArray(&r1, chain_num);
    DeleteArray(&r2, chain_num);
    DeleteArray(&xt, chain_num);
    return MMscore;
}

void copy_chain_data(const vector<vector<double> >&a_vec_i,
    const vector<char>&seq_vec_i,const vector<char>&sec_vec_i,
    const int len,double **a,char *seq,char *sec)
{
    int r;
    for (r=0;r<len;r++)
    {
        a[r][0]=a_vec_i[r][0];
        a[r][1]=a_vec_i[r][1];
        a[r][2]=a_vec_i[r][2];
        seq[r]=seq_vec_i[r];
        sec[r]=sec_vec_i[r];
    }
    seq[len]=0;
    sec[len]=0;
}

/* clear chains with L<3 */
void clear_full_PDB_lines(vector<vector<string> > PDB_lines,const string atom_opt)
{
    int chain_i;
    int Lch;
    int a;
    bool select_atom;
    string line;
    for (chain_i=0;chain_i<PDB_lines.size();chain_i++)
    {
        Lch=0;
        for (a=0;a<PDB_lines[chain_i].size();a++)
        {
            line=PDB_lines[chain_i][a];
            if (atom_opt=="auto")
            {
                if (line[17]==' ' && (line[18]=='D'||line[18]==' '))
                     select_atom=(line.compare(12,4," C3'")==0);
                else select_atom=(line.compare(12,4," CA ")==0);
            }
            else     select_atom=(line.compare(12,4,atom_opt)==0);
            Lch+=select_atom;
        }
        if (Lch<3)
        {
            for (a=0;a<PDB_lines[chain_i].size();a++)
                PDB_lines[chain_i][a].clear();
            PDB_lines[chain_i].clear();
        }
    }
    line.clear();
}

size_t get_full_PDB_lines(const string filename,
    vector<vector<string> >&PDB_lines, const int ter_opt,
    const int infmt_opt, const int split_opt, const int het_opt)
{
    size_t i=0; // resi i.e. atom index
    string line;
    char chainID=0;
    vector<string> tmp_str_vec;
    
    int compress_type=0; // uncompressed file
    ifstream fin;
#ifndef REDI_PSTREAM_H_SEEN
    ifstream fin_gz;
#else
    redi::ipstream fin_gz; // if file is compressed
    if (filename.size()>=3 && 
        filename.substr(filename.size()-3,3)==".gz")
    {
        fin_gz.open("gunzip -c '"+filename+"'");
        compress_type=1;
    }
    else if (filename.size()>=4 && 
        filename.substr(filename.size()-4,4)==".bz2")
    {
        fin_gz.open("bzcat '"+filename+"'");
        compress_type=2;
    }
    else
#endif
        fin.open(filename.c_str());

    if (infmt_opt==0||infmt_opt==-1) // PDB format
    {
        while (compress_type?fin_gz.good():fin.good())
        {
            if (compress_type) getline(fin_gz, line);
            else               getline(fin, line);
            if (infmt_opt==-1 && line.compare(0,5,"loop_")==0) // PDBx/mmCIF
                return get_full_PDB_lines(filename,PDB_lines,
                    ter_opt, 3, split_opt,het_opt);
            if (i > 0)
            {
                if      (ter_opt>=1 && line.compare(0,3,"END")==0) break;
                else if (ter_opt>=3 && line.compare(0,3,"TER")==0) break;
            }
            if (split_opt && line.compare(0,3,"END")==0) chainID=0;
            if (line.size()>=54 && (line[16]==' ' || line[16]=='A') && (
                (line.compare(0, 6, "ATOM  ")==0) || 
                (line.compare(0, 6, "HETATM")==0 && het_opt==1) ||
                (line.compare(0, 6, "HETATM")==0 && het_opt==2 && 
                 line.compare(17,3, "MSE")==0)))
            {
                if (!chainID)
                {
                    chainID=line[21];
                    PDB_lines.push_back(tmp_str_vec);
                }
                else if (ter_opt>=2 && chainID!=line[21]) break;
                if (split_opt==2 && chainID!=line[21])
                {
                    chainID=line[21];
                    PDB_lines.push_back(tmp_str_vec);
                } 
               
                PDB_lines.back().push_back(line);
                i++;
            }
        }
    }
    else if (infmt_opt==1) // SPICKER format
    {
        size_t L=0;
        float x,y,z;
        stringstream i8_stream;
        while (compress_type?fin_gz.good():fin.good())
        {
            if (compress_type) fin_gz>>L>>x>>y>>z;
            else               fin   >>L>>x>>y>>z;
            if (compress_type) getline(fin_gz, line);
            else               getline(fin, line);
            if (!(compress_type?fin_gz.good():fin.good())) break;
            for (i=0;i<L;i++)
            {
                if (compress_type) fin_gz>>x>>y>>z;
                else               fin   >>x>>y>>z;
                i8_stream<<"ATOM   "<<setw(4)<<i+1<<"  CA  UNK  "<<setw(4)
                    <<i+1<<"    "<<setiosflags(ios::fixed)<<setprecision(3)
                    <<setw(8)<<x<<setw(8)<<y<<setw(8)<<z;
                line=i8_stream.str();
                i8_stream.str(string());
                PDB_lines.back().push_back(line);
            }
            if (compress_type) getline(fin_gz, line);
            else               getline(fin, line);
        }
    }
    else if (infmt_opt==2) // xyz format
    {
        size_t L=0;
        stringstream i8_stream;
        while (compress_type?fin_gz.good():fin.good())
        {
            if (compress_type) getline(fin_gz, line);
            else               getline(fin, line);
            L=atoi(line.c_str());
            if (compress_type) getline(fin_gz, line);
            else               getline(fin, line);
            for (i=0;i<line.size();i++)
                if (line[i]==' '||line[i]=='\t') break;
            if (!(compress_type?fin_gz.good():fin.good())) break;
            PDB_lines.push_back(tmp_str_vec);
            for (i=0;i<L;i++)
            {
                if (compress_type) getline(fin_gz, line);
                else               getline(fin, line);
                i8_stream<<"ATOM   "<<setw(4)<<i+1<<"  CA  "
                    <<AAmap(line[0])<<"  "<<setw(4)<<i+1<<"    "
                    <<line.substr(2,8)<<line.substr(11,8)<<line.substr(20,8);
                line=i8_stream.str();
                i8_stream.str(string());
                PDB_lines.back().push_back(line);
            }
        }
    }
    else if (infmt_opt==3) // PDBx/mmCIF format
    {
        bool loop_ = false; // not reading following content
        map<string,int> _atom_site;
        int atom_site_pos;
        vector<string> line_vec;
        string alt_id=".";  // alternative location indicator
        string asym_id="."; // this is similar to chainID, except that
                            // chainID is char while asym_id is a string
                            // with possibly multiple char
        string prev_asym_id="";
        string AA="";       // residue name
        string atom="";
        string resi="";
        string model_index=""; // the same as model_idx but type is string
        stringstream i8_stream;
        while (compress_type?fin_gz.good():fin.good())
        {
            if (compress_type) getline(fin_gz, line);
            else               getline(fin, line);
            if (line.size()==0) continue;
            if (loop_) loop_ = line.compare(0,2,"# ");
            if (!loop_)
            {
                if (line.compare(0,5,"loop_")) continue;
                while(1)
                {
                    if (compress_type)
                    {
                        if (fin_gz.good()) getline(fin_gz, line);
                        else PrintErrorAndQuit("ERROR! Unexpected end of "+filename);
                    }
                    else
                    {
                        if (fin.good()) getline(fin, line);
                        else PrintErrorAndQuit("ERROR! Unexpected end of "+filename);
                    }
                    if (line.size()) break;
                }
                if (line.compare(0,11,"_atom_site.")) continue;

                loop_=true;
                _atom_site.clear();
                atom_site_pos=0;
                _atom_site[line.substr(11,line.size()-12)]=atom_site_pos;

                while(1)
                {
                    if (compress_type) getline(fin_gz, line);
                    else               getline(fin, line);
                    if (line.size()==0) continue;
                    if (line.compare(0,11,"_atom_site.")) break;
                    _atom_site[line.substr(11,line.size()-12)]=++atom_site_pos;
                }


                if (_atom_site.count("group_PDB")*
                    _atom_site.count("label_atom_id")*
                    _atom_site.count("label_comp_id")*
                   (_atom_site.count("auth_asym_id")+
                    _atom_site.count("label_asym_id"))*
                   (_atom_site.count("auth_seq_id")+
                    _atom_site.count("label_seq_id"))*
                    _atom_site.count("Cartn_x")*
                    _atom_site.count("Cartn_y")*
                    _atom_site.count("Cartn_z")==0)
                {
                    loop_ = false;
                    cerr<<"Warning! Missing one of the following _atom_site data items: group_PDB, label_atom_id, label_comp_id, auth_asym_id/label_asym_id, auth_seq_id/label_seq_id, Cartn_x, Cartn_y, Cartn_z"<<endl;
                    continue;
                }
            }

            line_vec.clear();
            split(line,line_vec);
            if ((line_vec[_atom_site["group_PDB"]]!="ATOM" &&
                 line_vec[_atom_site["group_PDB"]]!="HETATM") ||
                (line_vec[_atom_site["group_PDB"]]=="HETATM" &&
                 (het_opt==0 || 
                 (het_opt==2 && line_vec[_atom_site["label_comp_id"]]!="MSE")))
                ) continue;
            
            alt_id=".";
            if (_atom_site.count("label_alt_id")) // in 39.4 % of entries
                alt_id=line_vec[_atom_site["label_alt_id"]];
            if (alt_id!="." && alt_id!="A") continue;

            atom=line_vec[_atom_site["label_atom_id"]];
            if (atom[0]=='"') atom=atom.substr(1);
            if (atom.size() && atom[atom.size()-1]=='"')
                atom=atom.substr(0,atom.size()-1);
            if (atom.size()==0) continue;
            if      (atom.size()==1) atom=" "+atom+"  ";
            else if (atom.size()==2) atom=" "+atom+" "; // wrong for sidechain H
            else if (atom.size()==3) atom=" "+atom;
            else if (atom.size()>=5) continue;

            AA=line_vec[_atom_site["label_comp_id"]]; // residue name
            if      (AA.size()==1) AA="  "+AA;
            else if (AA.size()==2) AA=" " +AA;
            else if (AA.size()>=4) continue;

            if (_atom_site.count("auth_asym_id"))
                 asym_id=line_vec[_atom_site["auth_asym_id"]];
            else asym_id=line_vec[_atom_site["label_asym_id"]];
            if (asym_id==".") asym_id=" ";
            
            if (_atom_site.count("pdbx_PDB_model_num") && 
                model_index!=line_vec[_atom_site["pdbx_PDB_model_num"]])
            {
                model_index=line_vec[_atom_site["pdbx_PDB_model_num"]];
                if (PDB_lines.size() && ter_opt>=1) break;
                if (PDB_lines.size()==0 || split_opt>=1)
                {
                    PDB_lines.push_back(tmp_str_vec);
                    prev_asym_id=asym_id;
                }
            }

            if (prev_asym_id!=asym_id)
            {
                if (prev_asym_id!="" && ter_opt>=2) break;
                if (split_opt>=2) PDB_lines.push_back(tmp_str_vec);
            }
            if (prev_asym_id!=asym_id) prev_asym_id=asym_id;

            if (_atom_site.count("auth_seq_id"))
                 resi=line_vec[_atom_site["auth_seq_id"]];
            else resi=line_vec[_atom_site["label_seq_id"]];
            if (_atom_site.count("pdbx_PDB_ins_code") && 
                line_vec[_atom_site["pdbx_PDB_ins_code"]]!="?")
                resi+=line_vec[_atom_site["pdbx_PDB_ins_code"]][0];
            else resi+=" ";

            i++;
            i8_stream<<"ATOM  "
                <<setw(5)<<i<<" "<<atom<<" "<<AA<<setw(2)<<asym_id.substr(0,2)
                <<setw(5)<<resi.substr(0,5)<<"   "
                <<setw(8)<<line_vec[_atom_site["Cartn_x"]].substr(0,8)
                <<setw(8)<<line_vec[_atom_site["Cartn_y"]].substr(0,8)
                <<setw(8)<<line_vec[_atom_site["Cartn_z"]].substr(0,8);
            PDB_lines.back().push_back(i8_stream.str());
            i8_stream.str(string());
        }
        _atom_site.clear();
        line_vec.clear();
        alt_id.clear();
        asym_id.clear();
        AA.clear();
    }

    if (compress_type) fin_gz.close();
    else               fin.close();
    line.clear();
    return PDB_lines.size();
}

void output_dock(const vector<string>&chain_list, const int ter_opt,
    const int split_opt, const int infmt_opt, const string atom_opt,
    const int mirror_opt, double **ut_mat, const string&fname_super)
{
    size_t i;
    int chain_i,a;
    string name;
    int chainnum;
    double x[3];  // before transform
    double x1[3]; // after transform
    string line;
    vector<vector<string> >PDB_lines;
    int m=0;
    double t[3];
    double u[3][3];
    int ui,uj;
    stringstream buf;
    string filename;
    int het_opt=1;
    for (i=0;i<chain_list.size();i++)
    {
        name=chain_list[i];
        chainnum=get_full_PDB_lines(name, PDB_lines,
            ter_opt, infmt_opt, split_opt, het_opt);
        if (!chainnum) continue;
        clear_full_PDB_lines(PDB_lines, atom_opt); // clear chains with <3 residue
        for (chain_i=0;chain_i<chainnum;chain_i++)
        {
            if (PDB_lines[chain_i].size()<3) continue;
            buf<<fname_super<<'.'<<m<<".pdb";
            filename=buf.str();
            buf.str(string());
            for (ui=0;ui<3;ui++) for (uj=0;uj<3;uj++) u[ui][uj]=ut_mat[m][ui*3+uj];
            for (uj=0;uj<3;uj++) t[uj]=ut_mat[m][9+uj];
            for (a=0;a<PDB_lines[chain_i].size();a++)
            {
                line=PDB_lines[chain_i][a];
                x[0]=atof(line.substr(30,8).c_str());
                x[1]=atof(line.substr(38,8).c_str());
                x[2]=atof(line.substr(46,8).c_str());
                if (mirror_opt) x[2]=-x[2];
                transform(t, u, x, x1);
                buf<<line.substr(0,30)<<setiosflags(ios::fixed)
                   <<setprecision(3)
                   <<setw(8)<<x1[0]<<setw(8)<<x1[1]<<setw(8)<<x1[2]
                   <<line.substr(54)<<'\n';
            }
            buf<<"TER"<<endl;
            ofstream fp;
            fp.open(filename.c_str());
            fp<<buf.str();
            fp.close();
            buf.str(string());
            PDB_lines[chain_i].clear();
            m++;
        } // chain_i
        name.clear();
        PDB_lines.clear();
    } // i
    vector<vector<string> >().swap(PDB_lines);
    line.clear();
}

void parse_chain_list(const vector<string>&chain_list,
    vector<vector<vector<double> > >&a_vec, vector<vector<char> >&seq_vec,
    vector<vector<char> >&sec_vec, vector<int>&mol_vec, vector<int>&len_vec,
    vector<string>&chainID_list, const int ter_opt, const int split_opt,
    const string mol_opt, const int infmt_opt, const string atom_opt,
    const int mirror_opt, const int het_opt, int &len_aa, int &len_na,  
    const int o_opt, vector<string>&resi_vec)
{
    size_t i;
    int chain_i,r;
    string name;
    int chainnum;
    double **xa;
    int len;
    char *seq,*sec;

    vector<vector<string> >PDB_lines;
    vector<double> tmp_atom_array(3,0);
    vector<vector<double> > tmp_chain_array;
    vector<char>tmp_seq_array;
    vector<char>tmp_sec_array;
    //vector<string> resi_vec;
    int read_resi=2;

    for (i=0;i<chain_list.size();i++)
    {
        name=chain_list[i];
        chainnum=get_PDB_lines(name, PDB_lines, chainID_list,
            mol_vec, ter_opt, infmt_opt, atom_opt, split_opt, het_opt);
        if (!chainnum)
        {
            cerr<<"Warning! Cannot parse file: "<<name
                <<". Chain number 0."<<endl;
            continue;
        }
        for (chain_i=0;chain_i<chainnum;chain_i++)
        {
            len=PDB_lines[chain_i].size();
            if (!len)
            {
                cerr<<"Warning! Cannot parse file: "<<name
                    <<". Chain length 0."<<endl;
                continue;
            }
            else if (len<3)
            {
                cerr<<"Sequence is too short <3!: "<<name<<endl;
                continue;
            }
            NewArray(&xa, len, 3);
            seq = new char[len + 1];
            sec = new char[len + 1];
            len = read_PDB(PDB_lines[chain_i], xa, seq, resi_vec, read_resi);
            if (mirror_opt) for (r=0;r<len;r++) xa[r][2]=-xa[r][2];
            if (mol_vec[chain_i]>0 || mol_opt=="RNA")
                make_sec(seq, xa, len, sec,atom_opt);
            else make_sec(xa, len, sec); // secondary structure assignment
            
            /* store in vector */
            tmp_chain_array.assign(len,tmp_atom_array);
            vector<char>tmp_seq_array(len+1,0);
            vector<char>tmp_sec_array(len+1,0);
            for (r=0;r<len;r++)
            {
                tmp_chain_array[r][0]=xa[r][0];
                tmp_chain_array[r][1]=xa[r][1];
                tmp_chain_array[r][2]=xa[r][2];
                tmp_seq_array[r]=seq[r];
                tmp_sec_array[r]=sec[r];
            }
            a_vec.push_back(tmp_chain_array);
            seq_vec.push_back(tmp_seq_array);
            sec_vec.push_back(tmp_sec_array);
            len_vec.push_back(len);

            /* clean up */
            tmp_chain_array.clear();
            tmp_seq_array.clear();
            tmp_sec_array.clear();
            PDB_lines[chain_i].clear();
            DeleteArray(&xa, len);
            delete [] seq;
            delete [] sec;
        } // chain_i
        name.clear();
        PDB_lines.clear();
        mol_vec.clear();
    } // i
    tmp_atom_array.clear();

    if (mol_opt=="RNA") mol_vec.assign(a_vec.size(),1);
    else if (mol_opt=="protein") mol_vec.assign(a_vec.size(),-1);
    else
    {
        mol_vec.assign(a_vec.size(),0);
        for (i=0;i<a_vec.size();i++)
        {
            for (r=0;r<len_vec[i];r++)
            {
                if (seq_vec[i][r]>='a' && seq_vec[i][r]<='z') mol_vec[i]++;
                else mol_vec[i]--;
            }
        }
    }

    len_aa=0;
    len_na=0;
    for (i=0;i<a_vec.size();i++)
    {
        if (mol_vec[i]>0) len_na+=len_vec[i];
        else              len_aa+=len_vec[i];
    }
}

int copy_chain_pair_data(
    const vector<vector<vector<double> > >&xa_vec,
    const vector<vector<vector<double> > >&ya_vec,
    const vector<vector<char> >&seqx_vec, const vector<vector<char> >&seqy_vec,
    const vector<vector<char> >&secx_vec, const vector<vector<char> >&secy_vec,
    const vector<int> &mol_vec1, const vector<int> &mol_vec2,
    const vector<int> &xlen_vec, const vector<int> &ylen_vec,
    double **xa, double **ya, char *seqx, char *seqy, char *secx, char *secy,
    int chain1_num, int chain2_num,
    vector<vector<string> >&seqxA_mat, vector<vector<string> >&seqyA_mat,
    int *assign1_list, int *assign2_list, vector<string>&sequence)
{
    int i,j,r;
    for (i=0;i<sequence.size();i++) sequence[i].clear();
    sequence.clear();
    sequence.push_back("");
    sequence.push_back("");
    int mol_type=0;
    int xlen=0;
    int ylen=0;
    for (i=0;i<chain1_num;i++)
    {
        j=assign1_list[i];
        if (j<0) continue;
        for (r=0;r<xlen_vec[i];r++)
        {
            seqx[xlen]=seqx_vec[i][r];
            secx[xlen]=secx_vec[i][r];
            xa[xlen][0]= xa_vec[i][r][0];
            xa[xlen][1]= xa_vec[i][r][1];
            xa[xlen][2]= xa_vec[i][r][2];
            xlen++;
        }
        sequence[0]+=seqxA_mat[i][j];
        for (r=0;r<ylen_vec[j];r++)
        {
            seqy[ylen]=seqy_vec[j][r];
            secy[ylen]=secy_vec[j][r];
            ya[ylen][0]= ya_vec[j][r][0];
            ya[ylen][1]= ya_vec[j][r][1];
            ya[ylen][2]= ya_vec[j][r][2];
            ylen++;
        }
        sequence[1]+=seqyA_mat[i][j];
        mol_type+=mol_vec1[i]+mol_vec2[j];
    }
    seqx[xlen]=0;
    secx[xlen]=0;
    seqy[ylen]=0;
    secy[ylen]=0;
    return mol_type;
}

double MMalign_search(
    const vector<vector<vector<double> > >&xa_vec,
    const vector<vector<vector<double> > >&ya_vec,
    const vector<vector<char> >&seqx_vec, const vector<vector<char> >&seqy_vec,
    const vector<vector<char> >&secx_vec, const vector<vector<char> >&secy_vec,
    const vector<int> &mol_vec1, const vector<int> &mol_vec2,
    const vector<int> &xlen_vec, const vector<int> &ylen_vec,
    double **xa, double **ya, char *seqx, char *seqy, char *secx, char *secy,
    int len_aa, int len_na, int chain1_num, int chain2_num, double **TMave_mat,
    vector<vector<string> >&seqxA_mat, vector<vector<string> >&seqyA_mat,
    int *assign1_list, int *assign2_list, vector<string>&sequence,
    double d0_scale, bool fast_opt, const int i_opt=3)
{
    double total_score=0;
    int i,j;
    int xlen=0;
    int ylen=0;
    for (i=0;i<chain1_num;i++)
    {
        if (assign1_list[i]<0) continue;
        xlen+=xlen_vec[i];
        ylen+=ylen_vec[assign1_list[i]];
    }
    if (xlen<=3 || ylen<=3) return total_score;

    seqx = new char[xlen+1];
    secx = new char[xlen+1];
    NewArray(&xa, xlen, 3);
    seqy = new char[ylen+1];
    secy = new char[ylen+1];
    NewArray(&ya, ylen, 3);

    int mol_type=copy_chain_pair_data(xa_vec, ya_vec, seqx_vec, seqy_vec,
        secx_vec, secy_vec, mol_vec1, mol_vec2, xlen_vec, ylen_vec,
        xa, ya, seqx, seqy, secx, secy, chain1_num, chain2_num,
        seqxA_mat, seqyA_mat, assign1_list, assign2_list, sequence);

    /* declare variable specific to this pair of TMalign */
    double t0[3], u0[3][3];
    double TM1, TM2;
    double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
    double d0_0, TM_0;
    double d0A, d0B, d0u, d0a;
    double d0_out=5.0;
    string seqM, seqxA, seqyA;// for output alignment
    double rmsd0 = 0.0;
    int L_ali;                // Aligned length in standard_TMscore
    double Liden=0;
    double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
    int n_ali=0;
    int n_ali8=0;

    double Lnorm_ass=len_aa+len_na;

    /* entry function for structure alignment */
    TMalign_main(xa, ya, seqx, seqy, secx, secy,
        t0, u0, TM1, TM2, TM3, TM4, TM5,
        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
        xlen, ylen, sequence, Lnorm_ass, d0_scale,
        i_opt, false, true, false, fast_opt, mol_type, -1);

    /* clean up */
    delete [] seqx;
    delete [] seqy;
    delete [] secx;
    delete [] secy;
    DeleteArray(&xa,xlen);
    DeleteArray(&ya,ylen);

    /* re-compute chain level alignment */
    for (i=0;i<chain1_num;i++)
    {
        xlen=xlen_vec[i];
        if (xlen<3)
        {
            for (j=0;j<chain2_num;j++) TMave_mat[i][j]=-1;
            continue;
        }
        seqx = new char[xlen+1];
        secx = new char[xlen+1];
        NewArray(&xa, xlen, 3);
        copy_chain_data(xa_vec[i],seqx_vec[i],secx_vec[i],
            xlen,xa,seqx,secx);

        double **xt;
        NewArray(&xt, xlen, 3);
        do_rotation(xa, xt, xlen, t0, u0);

        for (j=0;j<chain2_num;j++)
        {
            if (mol_vec1[i]*mol_vec2[j]<0) //no protein-RNA alignment
            {
                TMave_mat[i][j]=-1;
                continue;
            }

            ylen=ylen_vec[j];
            if (ylen<3)
            {
                TMave_mat[i][j]=-1;
                continue;
            }
            seqy = new char[ylen+1];
            secy = new char[ylen+1];
            NewArray(&ya, ylen, 3);
            copy_chain_data(ya_vec[j],seqy_vec[j],secy_vec[j],
                ylen,ya,seqy,secy);

            /* declare variable specific to this pair of TMalign */
            d0_out=5.0;
            seqM.clear();
            seqxA.clear();
            seqyA.clear();
            rmsd0 = 0.0;
            Liden=0;
            int *invmap = new int[ylen+1];

            double Lnorm_ass=len_aa;
            if (mol_vec1[i]+mol_vec2[j]>0) Lnorm_ass=len_na;

            /* entry function for structure alignment */
            se_main(xt, ya, seqx, seqy, TM1, TM2, TM3, TM4, TM5,
                d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
                rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                xlen, ylen, sequence, Lnorm_ass, d0_scale,
                0, false, 2, false, mol_vec1[i]+mol_vec2[j], 1, invmap);

            /* print result */
            seqxA_mat[i][j]=seqxA;
            seqyA_mat[i][j]=seqyA;

            TMave_mat[i][j]=TM4*Lnorm_ass;
            if (assign1_list[i]==j) total_score+=TMave_mat[i][j];

            /* clean up */
            seqM.clear();
            seqxA.clear();
            seqyA.clear();

            delete[]seqy;
            delete[]secy;
            DeleteArray(&ya,ylen);
            delete[]invmap;
        }
        delete[]seqx;
        delete[]secx;
        DeleteArray(&xa,xlen);
        DeleteArray(&xt,xlen);
    }
    return total_score;
}

void MMalign_final(
    const string xname, const string yname,
    const vector<string> chainID_list1, const vector<string> chainID_list2,
    string fname_super, string fname_lign, string fname_matrix,
    const vector<vector<vector<double> > >&xa_vec,
    const vector<vector<vector<double> > >&ya_vec,
    const vector<vector<char> >&seqx_vec, const vector<vector<char> >&seqy_vec,
    const vector<vector<char> >&secx_vec, const vector<vector<char> >&secy_vec,
    const vector<int> &mol_vec1, const vector<int> &mol_vec2,
    const vector<int> &xlen_vec, const vector<int> &ylen_vec,
    double **xa, double **ya, char *seqx, char *seqy, char *secx, char *secy,
    int len_aa, int len_na, int chain1_num, int chain2_num,
    double **TMave_mat,
    vector<vector<string> >&seqxA_mat, vector<vector<string> >&seqM_mat,
    vector<vector<string> >&seqyA_mat, int *assign1_list, int *assign2_list,
    vector<string>&sequence, const double d0_scale, const bool m_opt,
    const int o_opt, const int outfmt_opt, const int ter_opt,
    const int split_opt, const bool a_opt, const bool d_opt,
    const bool fast_opt, const bool full_opt, const int mirror_opt,
    const vector<string>&resi_vec1, const vector<string>&resi_vec2)
{
    int i,j;
    int xlen=0;
    int ylen=0;
    for (i=0;i<chain1_num;i++) xlen+=xlen_vec[i];
    for (j=0;j<chain2_num;j++) ylen+=ylen_vec[j];
    if (xlen<=3 || ylen<=3) return;

    seqx = new char[xlen+1];
    secx = new char[xlen+1];
    NewArray(&xa, xlen, 3);
    seqy = new char[ylen+1];
    secy = new char[ylen+1];
    NewArray(&ya, ylen, 3);

    int mol_type=copy_chain_pair_data(xa_vec, ya_vec, seqx_vec, seqy_vec,
        secx_vec, secy_vec, mol_vec1, mol_vec2, xlen_vec, ylen_vec,
        xa, ya, seqx, seqy, secx, secy, chain1_num, chain2_num,
        seqxA_mat, seqyA_mat, assign1_list, assign2_list, sequence);

    /* declare variable specific to this pair of TMalign */
    double t0[3], u0[3][3];
    double TM1, TM2;
    double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
    double d0_0, TM_0;
    double d0A, d0B, d0u, d0a;
    double d0_out=5.0;
    string seqM, seqxA, seqyA;// for output alignment
    double rmsd0 = 0.0;
    int L_ali;                // Aligned length in standard_TMscore
    double Liden=0;
    double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
    int n_ali=0;
    int n_ali8=0;

    double Lnorm_ass=len_aa+len_na;

    /* entry function for structure alignment */
    TMalign_main(xa, ya, seqx, seqy, secx, secy,
        t0, u0, TM1, TM2, TM3, TM4, TM5,
        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
        xlen, ylen, sequence, Lnorm_ass, d0_scale,
        3, a_opt, false, d_opt, fast_opt, mol_type, -1);

    /* prepare full complex alignment */
    string chainID1="";
    string chainID2="";
    sequence.clear();
    sequence.push_back(""); // seqxA
    sequence.push_back(""); // seqyA
    sequence.push_back(""); // seqM
    int aln_start=0;
    int aln_end=0;
    for (i=0;i<chain1_num;i++)
    {
        j=assign1_list[i];
        if (j<0) continue;
        chainID1+=chainID_list1[i];
        chainID2+=chainID_list2[j];
        sequence[0]+=seqxA_mat[i][j]+'*';
        sequence[1]+=seqyA_mat[i][j]+'*';

        aln_end+=seqxA_mat[i][j].size();
        seqM_mat[i][j]=seqM.substr(aln_start,aln_end-aln_start);
        sequence[2]+=seqM_mat[i][j]+'*';
        aln_start=aln_end;
    }

    /* prepare unaligned region */
    for (i=0;i<chain1_num;i++)
    {
        if (assign1_list[i]>=0) continue;
        chainID1+=chainID_list1[i];
        chainID2+=':';
        string s(seqx_vec[i].begin(),seqx_vec[i].end());
        sequence[0]+=s.substr(0,xlen_vec[i])+'*';
        sequence[1]+=string(xlen_vec[i],'-')+'*';
        s.clear();
        sequence[2]+=string(xlen_vec[i],' ')+'*';
    }
    for (j=0;j<chain2_num;j++)
    {
        if (assign2_list[j]>=0) continue;
        chainID1+=':';
        chainID2+=chainID_list2[j];
        string s(seqy_vec[j].begin(),seqy_vec[j].end());
        sequence[0]+=string(ylen_vec[j],'-')+'*';
        sequence[1]+=s.substr(0,ylen_vec[j])+'*';
        s.clear();
        sequence[2]+=string(ylen_vec[j],' ')+'*';
    }

    /* print alignment */
    output_results(xname, yname, chainID1.c_str(), chainID2.c_str(),
        xlen, ylen, t0, u0, TM1, TM2, TM3, TM4, TM5, rmsd0, d0_out,
        sequence[2].c_str(), sequence[0].c_str(), sequence[1].c_str(),
        Liden, n_ali8, L_ali, TM_ali, rmsd_ali,
        TM_0, d0_0, d0A, d0B, 0, d0_scale, d0a, d0u, 
        (m_opt?fname_matrix:"").c_str(), outfmt_opt, ter_opt, true,
        split_opt, o_opt, fname_super,
        false, a_opt, false, d_opt, mirror_opt, resi_vec1, resi_vec2);

    /* clean up */
    seqM.clear();
    seqxA.clear();
    seqyA.clear();
    delete [] seqx;
    delete [] seqy;
    delete [] secx;
    delete [] secy;
    DeleteArray(&xa,xlen);
    DeleteArray(&ya,ylen);
    sequence[0].clear();
    sequence[1].clear();
    sequence[2].clear();

    if (!full_opt) return;

    cout<<"# End of alignment for full complex. The following blocks list alignments for individual chains."<<endl;

    /* re-compute chain level alignment */
    for (i=0;i<chain1_num;i++)
    {
        j=assign1_list[i];
        if (j<0) continue;
        xlen=xlen_vec[i];
        seqx = new char[xlen+1];
        secx = new char[xlen+1];
        NewArray(&xa, xlen, 3);
        copy_chain_data(xa_vec[i],seqx_vec[i],secx_vec[i],
            xlen,xa,seqx,secx);

        double **xt;
        NewArray(&xt, xlen, 3);
        do_rotation(xa, xt, xlen, t0, u0);

        ylen=ylen_vec[j];
        if (ylen<3)
        {
            TMave_mat[i][j]=-1;
            continue;
        }
        seqy = new char[ylen+1];
        secy = new char[ylen+1];
        NewArray(&ya, ylen, 3);
        copy_chain_data(ya_vec[j],seqy_vec[j],secy_vec[j],
            ylen,ya,seqy,secy);

        /* declare variable specific to this pair of TMalign */
        d0_out=5.0;
        rmsd0 = 0.0;
        Liden=0;
        int *invmap = new int[ylen+1];
        seqM="";
        seqxA="";
        seqyA="";
        double Lnorm_ass=len_aa;
        if (mol_vec1[i]+mol_vec2[j]>0) Lnorm_ass=len_na;
        sequence[0]=seqxA_mat[i][j];
        sequence[1]=seqyA_mat[i][j];

        /* entry function for structure alignment */
        se_main(xt, ya, seqx, seqy, TM1, TM2, TM3, TM4, TM5,
            d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
            rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
            xlen, ylen, sequence, Lnorm_ass, d0_scale,
            1, a_opt, 2, d_opt, mol_vec1[i]+mol_vec2[j], 1, invmap);

        //TM2=TM4*Lnorm_ass/xlen;
        //TM1=TM4*Lnorm_ass/ylen;
        //d0A=d0u;
        //d0B=d0u;

        /* print result */
        output_results(xname, yname,
            chainID_list1[i].c_str(), chainID_list2[j].c_str(),
            xlen, ylen, t0, u0, TM1, TM2, TM3, TM4, TM5, rmsd0, d0_out,
            seqM_mat[i][j].c_str(), seqxA_mat[i][j].c_str(),
            seqyA_mat[i][j].c_str(), Liden, n_ali8, L_ali, TM_ali, rmsd_ali,
            TM_0, d0_0, d0A, d0B, Lnorm_ass, d0_scale, d0a, d0u, 
            "", outfmt_opt, ter_opt, false, split_opt, 0,
            "", false, a_opt, false, d_opt, 0, resi_vec1, resi_vec2);

        /* clean up */
        seqxA.clear();
        seqM.clear();
        seqyA.clear();
        sequence[0].clear();
        sequence[1].clear();
        delete[]seqy;
        delete[]secy;
        DeleteArray(&ya,ylen);
        delete[]seqx;
        delete[]secx;
        DeleteArray(&xa,xlen);
        DeleteArray(&xt,xlen);
        delete[]invmap;
    }
    sequence.clear();
    return;
}

void copy_chain_assign_data(int chain1_num, int chain2_num, 
    vector<string> &sequence,
    vector<vector<string> >&seqxA_mat, vector<vector<string> >&seqyA_mat,
    int *assign1_list, int *assign2_list, double **TMave_mat,
    vector<vector<string> >&seqxA_tmp, vector<vector<string> >&seqyA_tmp,
    int *assign1_tmp,  int *assign2_tmp,  double **TMave_tmp)
{
    int i,j;
    for (i=0;i<sequence.size();i++) sequence[i].clear();
    sequence.clear();
    sequence.push_back("");
    sequence.push_back("");
    for (i=0;i<chain1_num;i++) assign1_tmp[i]=assign1_list[i];
    for (i=0;i<chain2_num;i++) assign2_tmp[i]=assign2_list[i];
    for (i=0;i<chain1_num;i++)
    {
        for (j=0;j<chain2_num;j++)
        {
            seqxA_tmp[i][j]=seqxA_mat[i][j];
            seqyA_tmp[i][j]=seqyA_mat[i][j];
            TMave_tmp[i][j]=TMave_mat[i][j];
            if (assign1_list[i]==j)
            {
                sequence[0]+=seqxA_mat[i][j];
                sequence[1]+=seqyA_mat[i][j];
            }
        }
    }
    return;
}

void MMalign_iter(double & max_total_score, const int max_iter,
    const vector<vector<vector<double> > >&xa_vec,
    const vector<vector<vector<double> > >&ya_vec,
    const vector<vector<char> >&seqx_vec, const vector<vector<char> >&seqy_vec,
    const vector<vector<char> >&secx_vec, const vector<vector<char> >&secy_vec,
    const vector<int> &mol_vec1, const vector<int> &mol_vec2,
    const vector<int> &xlen_vec, const vector<int> &ylen_vec,
    double **xa, double **ya, char *seqx, char *seqy, char *secx, char *secy,
    int len_aa, int len_na, int chain1_num, int chain2_num, double **TMave_mat,
    vector<vector<string> >&seqxA_mat, vector<vector<string> >&seqyA_mat,
    int *assign1_list, int *assign2_list, vector<string>&sequence,
    double d0_scale, bool fast_opt)
{
    /* tmp assignment */
    double total_score;
    int *assign1_tmp, *assign2_tmp;
    assign1_tmp=new int[chain1_num];
    assign2_tmp=new int[chain2_num];
    double **TMave_tmp;
    NewArray(&TMave_tmp,chain1_num,chain2_num);
    vector<string> tmp_str_vec(chain2_num,"");
    vector<vector<string> >seqxA_tmp(chain1_num,tmp_str_vec);
    vector<vector<string> >seqyA_tmp(chain1_num,tmp_str_vec);
    vector<string> sequence_tmp;
    copy_chain_assign_data(chain1_num, chain2_num, sequence_tmp,
        seqxA_mat, seqyA_mat, assign1_list, assign2_list, TMave_mat,
        seqxA_tmp, seqyA_tmp, assign1_tmp,  assign2_tmp,  TMave_tmp);
    
    for (int iter=0;iter<max_iter;iter++)
    {
        total_score=MMalign_search(xa_vec, ya_vec, seqx_vec, seqy_vec,
            secx_vec, secy_vec, mol_vec1, mol_vec2, xlen_vec, ylen_vec,
            xa, ya, seqx, seqy, secx, secy, len_aa, len_na,
            chain1_num, chain2_num, 
            TMave_tmp, seqxA_tmp, seqyA_tmp, assign1_tmp, assign2_tmp,
            sequence, d0_scale, fast_opt);
        total_score=enhanced_greedy_search(TMave_tmp, assign1_tmp,
            assign2_tmp, chain1_num, chain2_num);
        //if (total_score<=0) PrintErrorAndQuit("ERROR! No assignable chain");
        if (total_score<=max_total_score) break;
        max_total_score=total_score;
        copy_chain_assign_data(chain1_num, chain2_num, sequence,
            seqxA_tmp, seqyA_tmp, assign1_tmp,  assign2_tmp,  TMave_tmp,
            seqxA_mat, seqyA_mat, assign1_list, assign2_list, TMave_mat);
    }

    /* clean up everything */
    delete [] assign1_tmp;
    delete [] assign2_tmp;
    DeleteArray(&TMave_tmp,chain1_num);
    vector<string>().swap(tmp_str_vec);
    vector<vector<string> >().swap(seqxA_tmp);
    vector<vector<string> >().swap(seqyA_tmp);
}


/* Input: vectors x, y, rotation matrix t, u, scale factor d02, and gap_open
 * Output: j2i[1:len2] \in {1:len1} U {-1}
 * path[0:len1, 0:len2]=1,2,3, from diagonal, horizontal, vertical */
void NWDP_TM_dimer(bool **path, double **val, double **x, double **y,
    int len1, int len2, bool **mask,
    double t[3], double u[3][3], double d02, double gap_open, int j2i[])
{
    int i, j;
    double h, v, d;

    //initialization
    for(i=0; i<=len1; i++)
    {
        //val[i][0]=0;
        val[i][0]=i*gap_open;
        path[i][0]=false; //not from diagonal
    }

    for(j=0; j<=len2; j++)
    {
        //val[0][j]=0;
        val[0][j]=j*gap_open;
        path[0][j]=false; //not from diagonal
        j2i[j]=-1;    //all are not aligned, only use j2i[1:len2]
    }      
    double xx[3], dij;


    //decide matrix and path
    for(i=1; i<=len1; i++)
    {
        transform(t, u, &x[i-1][0], xx);
        for(j=1; j<=len2; j++)
        {
            d=FLT_MIN;
            if (mask[i][j])
            {
                dij=dist(xx, &y[j-1][0]);    
                d=val[i-1][j-1] +  1.0/(1+dij/d02);
            } 

            //symbol insertion in horizontal (= a gap in vertical)
            h=val[i-1][j];
            if(path[i-1][j]) h += gap_open; //aligned in last position

            //symbol insertion in vertical
            v=val[i][j-1];
            if(path[i][j-1]) v += gap_open; //aligned in last position


            if(d>=h && d>=v)
            {
                path[i][j]=true; //from diagonal
                val[i][j]=d;
            }
            else 
            {
                path[i][j]=false; //from horizontal
                if(v>=h) val[i][j]=v;
                else val[i][j]=h;
            }
        } //for i
    } //for j

    //trace back to extract the alignment
    i=len1;
    j=len2;
    while(i>0 && j>0)
    {
        if(path[i][j]) //from diagonal
        {
            j2i[j-1]=i-1;
            i--;
            j--;
        }
        else 
        {
            h=val[i-1][j];
            if(path[i-1][j]) h +=gap_open;

            v=val[i][j-1];
            if(path[i][j-1]) v +=gap_open;

            if(v>=h) j--;
            else i--;
        }
    }
}

/* +ss
 * Input: secondary structure secx, secy, and gap_open
 * Output: j2i[1:len2] \in {1:len1} U {-1}
 * path[0:len1, 0:len2]=1,2,3, from diagonal, horizontal, vertical */
void NWDP_TM_dimer(bool **path, double **val, const char *secx, const char *secy,
    const int len1, const int len2, bool **mask, const double gap_open, int j2i[])
{

    int i, j;
    double h, v, d;

    //initialization
    for(i=0; i<=len1; i++)
    {
        //val[i][0]=0;
        val[i][0]=i*gap_open;
        path[i][0]=false; //not from diagonal
    }

    for(j=0; j<=len2; j++)
    {
        //val[0][j]=0;
        val[0][j]=j*gap_open;
        path[0][j]=false; //not from diagonal
        j2i[j]=-1;    //all are not aligned, only use j2i[1:len2]
    }      

    //decide matrix and path
    for(i=1; i<=len1; i++)
    {
        for(j=1; j<=len2; j++)
        {
            d=FLT_MIN;
            if (mask[i][j])
                d=val[i-1][j-1] + 1.0*(secx[i-1]==secy[j-1]);

            //symbol insertion in horizontal (= a gap in vertical)
            h=val[i-1][j];
            if(path[i-1][j]) h += gap_open; //aligned in last position

            //symbol insertion in vertical
            v=val[i][j-1];
            if(path[i][j-1]) v += gap_open; //aligned in last position

            if(d>=h && d>=v)
            {
                path[i][j]=true; //from diagonal
                val[i][j]=d;
            }
            else 
            {
                path[i][j]=false; //from horizontal
                if(v>=h) val[i][j]=v;
                else val[i][j]=h;
            }
        } //for i
    } //for j

    //trace back to extract the alignment
    i=len1;
    j=len2;
    while(i>0 && j>0)
    {
        if(path[i][j]) //from diagonal
        {
            j2i[j-1]=i-1;
            i--;
            j--;
        }
        else 
        {
            h=val[i-1][j];
            if(path[i-1][j]) h +=gap_open;

            v=val[i][j-1];
            if(path[i][j-1]) v +=gap_open;

            if(v>=h) j--;
            else i--;
        }
    }
}

//heuristic run of dynamic programing iteratively to find the best alignment
//input: initial rotation matrix t, u
//       vectors x and y, d0
//output: best alignment that maximizes the TMscore, will be stored in invmap
double DP_iter_dimer(double **r1, double **r2, double **xtm, double **ytm,
    double **xt, bool **path, double **val, double **x, double **y,
    int xlen, int ylen, bool **mask, double t[3], double u[3][3], int invmap0[],
    int g1, int g2, int iteration_max, double local_d0_search,
    double D0_MIN, double Lnorm, double d0, double score_d8)
{
    double gap_open[2]={-0.6, 0};
    double rmsd; 
    int *invmap=new int[ylen+1];
    
    int iteration, i, j, k;
    double tmscore, tmscore_max, tmscore_old=0;    
    int score_sum_method=8, simplify_step=40;
    tmscore_max=-1;

    //double d01=d0+1.5;
    double d02=d0*d0;
    for(int g=g1; g<g2; g++)
    {
        for(iteration=0; iteration<iteration_max; iteration++)
        {           
            NWDP_TM_dimer(path, val, x, y, xlen, ylen, mask,
                t, u, d02, gap_open[g], invmap);
            
            k=0;
            for(j=0; j<ylen; j++) 
            {
                i=invmap[j];

                if(i>=0) //aligned
                {
                    xtm[k][0]=x[i][0];
                    xtm[k][1]=x[i][1];
                    xtm[k][2]=x[i][2];
                    
                    ytm[k][0]=y[j][0];
                    ytm[k][1]=y[j][1];
                    ytm[k][2]=y[j][2];
                    k++;
                }
            }

            tmscore = TMscore8_search(r1, r2, xtm, ytm, xt, k, t, u,
                simplify_step, score_sum_method, &rmsd, local_d0_search,
                Lnorm, score_d8, d0);

           
            if(tmscore>tmscore_max)
            {
                tmscore_max=tmscore;
                for(i=0; i<ylen; i++) invmap0[i]=invmap[i];
            }
    
            if(iteration>0)
            {
                if(fabs(tmscore_old-tmscore)<0.000001) break;       
            }
            tmscore_old=tmscore;
        }// for iteration           
        
    }//for gapopen
    
    
    delete []invmap;
    return tmscore_max;
}

void get_initial_ss_dimer(bool **path, double **val, const char *secx,
    const char *secy, int xlen, int ylen, bool **mask, int *y2x)
{
    double gap_open=-1.0;
    NWDP_TM_dimer(path, val, secx, secy, xlen, ylen, mask, gap_open, y2x);
}

bool get_initial5_dimer( double **r1, double **r2, double **xtm, double **ytm,
    bool **path, double **val, double **x, double **y, int xlen, int ylen,
    bool **mask, int *y2x,
    double d0, double d0_search, const bool fast_opt, const double D0_MIN)
{
    double GL, rmsd;
    double t[3];
    double u[3][3];

    double d01 = d0 + 1.5;
    if (d01 < D0_MIN) d01 = D0_MIN;
    double d02 = d01*d01;

    double GLmax = 0;
    int aL = getmin(xlen, ylen);
    int *invmap = new int[ylen + 1];

    // jump on sequence1-------------->
    int n_jump1 = 0;
    if (xlen > 250)
        n_jump1 = 45;
    else if (xlen > 200)
        n_jump1 = 35;
    else if (xlen > 150)
        n_jump1 = 25;
    else
        n_jump1 = 15;
    if (n_jump1 > (xlen / 3))
        n_jump1 = xlen / 3;

    // jump on sequence2-------------->
    int n_jump2 = 0;
    if (ylen > 250)
        n_jump2 = 45;
    else if (ylen > 200)
        n_jump2 = 35;
    else if (ylen > 150)
        n_jump2 = 25;
    else
        n_jump2 = 15;
    if (n_jump2 > (ylen / 3))
        n_jump2 = ylen / 3;

    // fragment to superimpose-------------->
    int n_frag[2] = { 20, 100 };
    if (n_frag[0] > (aL / 3))
        n_frag[0] = aL / 3;
    if (n_frag[1] > (aL / 2))
        n_frag[1] = aL / 2;

    // start superimpose search-------------->
    if (fast_opt)
    {
        n_jump1*=5;
        n_jump2*=5;
    }
    bool flag = false;
    for (int i_frag = 0; i_frag < 2; i_frag++)
    {
        int m1 = xlen - n_frag[i_frag] + 1;
        int m2 = ylen - n_frag[i_frag] + 1;

        for (int i = 0; i<m1; i = i + n_jump1) //index starts from 0, different from FORTRAN
        {
            for (int j = 0; j<m2; j = j + n_jump2)
            {
                for (int k = 0; k<n_frag[i_frag]; k++) //fragment in y
                {
                    r1[k][0] = x[k + i][0];
                    r1[k][1] = x[k + i][1];
                    r1[k][2] = x[k + i][2];

                    r2[k][0] = y[k + j][0];
                    r2[k][1] = y[k + j][1];
                    r2[k][2] = y[k + j][2];
                }

                // superpose the two structures and rotate it
                Kabsch(r1, r2, n_frag[i_frag], 1, &rmsd, t, u);

                double gap_open = 0.0;
                NWDP_TM_dimer(path, val, x, y, xlen, ylen, mask,
                    t, u, d02, gap_open, invmap);
                GL = get_score_fast(r1, r2, xtm, ytm, x, y, xlen, ylen,
                    invmap, d0, d0_search, t, u);
                if (GL>GLmax)
                {
                    GLmax = GL;
                    for (int ii = 0; ii<ylen; ii++) y2x[ii] = invmap[ii];
                    flag = true;
                }
            }
        }
    }

    delete[] invmap;
    return flag;
}

void get_initial_ssplus_dimer(double **r1, double **r2, double **score,
    bool **path, double **val, const char *secx, const char *secy,
    double **x, double **y, int xlen, int ylen, bool **mask,
    int *y2x0, int *y2x, const double D0_MIN, double d0)
{
    //create score matrix for DP
    score_matrix_rmsd_sec(r1, r2, score, secx, secy, x, y, xlen, ylen,
        y2x0, D0_MIN,d0);

    int i,j;
    for (i=0;i<xlen+1;i++) for (j=0;j<ylen+1;j++) score[i][j]=FLT_MIN;
    
    double gap_open=-1.0;
    NWDP_TM(score, path, val, xlen, ylen, gap_open, y2x);
}

/* Entry function for TM-align. Return TM-score calculation status:
 * 0   - full TM-score calculation 
 * 1   - terminated due to exception
 * 2-7 - pre-terminated due to low TM-score */
int TMalign_dimer_main(double **xa, double **ya,
    const char *seqx, const char *seqy, const char *secx, const char *secy,
    double t0[3], double u0[3][3],
    double &TM1, double &TM2, double &TM3, double &TM4, double &TM5,
    double &d0_0, double &TM_0,
    double &d0A, double &d0B, double &d0u, double &d0a, double &d0_out,
    string &seqM, string &seqxA, string &seqyA,
    double &rmsd0, int &L_ali, double &Liden,
    double &TM_ali, double &rmsd_ali, int &n_ali, int &n_ali8,
    const int xlen, const int ylen,
    bool **mask,
    const vector<string> sequence, const double Lnorm_ass,
    const double d0_scale, const int i_opt, const int a_opt,
    const bool u_opt, const bool d_opt, const bool fast_opt,
    const int mol_type, const double TMcut=-1)
{
    double D0_MIN;        //for d0
    double Lnorm;         //normalization length
    double score_d8,d0,d0_search,dcu0;//for TMscore search
    double t[3], u[3][3]; //Kabsch translation vector and rotation matrix
    double **score;       // Input score table for dynamic programming
    bool   **path;        // for dynamic programming  
    double **val;         // for dynamic programming  
    double **xtm, **ytm;  // for TMscore search engine
    double **xt;          //for saving the superposed version of r_1 or xtm
    double **r1, **r2;    // for Kabsch rotation

    /***********************/
    /* allocate memory     */
    /***********************/
    int minlen = min(xlen, ylen);
    NewArray(&score, xlen+1, ylen+1);
    NewArray(&path, xlen+1, ylen+1);
    NewArray(&val, xlen+1, ylen+1);
    NewArray(&xtm, minlen, 3);
    NewArray(&ytm, minlen, 3);
    NewArray(&xt, xlen, 3);
    NewArray(&r1, minlen, 3);
    NewArray(&r2, minlen, 3);

    /***********************/
    /*    parameter set    */
    /***********************/
    parameter_set4search(xlen, ylen, D0_MIN, Lnorm, 
        score_d8, d0, d0_search, dcu0);
    int simplify_step    = 40; //for simplified search engine
    int score_sum_method = 8;  //for scoring method, whether only sum over pairs with dis<score_d8

    int i;
    int *invmap0         = new int[ylen+1];
    int *invmap          = new int[ylen+1];
    double TM, TMmax=-1;
    for(i=0; i<ylen; i++) invmap0[i]=-1;

    double ddcc=0.4;
    if (Lnorm <= 40) ddcc=0.1;   //Lnorm was setted in parameter_set4search
    double local_d0_search = d0_search;

    //************************************************//
    //    get initial alignment from user's input:    //
    //    Stick to the initial alignment              //
    //************************************************//
    bool bAlignStick = false;
    if (i_opt==3)// if input has set parameter for "-I"
    {
        // In the original code, this loop starts from 1, which is
        // incorrect. Fortran starts from 1 but C++ should starts from 0.
        for (int j = 0; j < ylen; j++)// Set aligned position to be "-1"
            invmap[j] = -1;

        int i1 = -1;// in C version, index starts from zero, not from one
        int i2 = -1;
        int L1 = sequence[0].size();
        int L2 = sequence[1].size();
        int L = min(L1, L2);// Get positions for aligned residues
        for (int kk1 = 0; kk1 < L; kk1++)
        {
            if (sequence[0][kk1] != '-') i1++;
            if (sequence[1][kk1] != '-')
            {
                i2++;
                if (i2 >= ylen || i1 >= xlen) kk1 = L;
                else if (sequence[0][kk1] != '-') invmap[i2] = i1;
            }
        }

        //--------------- 2. Align proteins from original alignment
        double prevD0_MIN = D0_MIN;// stored for later use
        int prevLnorm = Lnorm;
        double prevd0 = d0;
        TM_ali = standard_TMscore(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen,
            invmap, L_ali, rmsd_ali, D0_MIN, Lnorm, d0, d0_search, score_d8,
            t, u, mol_type);
        D0_MIN = prevD0_MIN;
        Lnorm = prevLnorm;
        d0 = prevd0;
        TM = detailed_search_standard(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen,
            invmap, t, u, 40, 8, local_d0_search, true, Lnorm, score_d8, d0);
        if (TM > TMmax)
        {
            TMmax = TM;
            for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
        }
        bAlignStick = true;
    }

    /******************************************************/
    /*    get initial alignment with gapless threading    */
    /******************************************************/
    if (!bAlignStick)
    {
        get_initial(r1, r2, xtm, ytm, xa, ya, xlen, ylen, invmap0, d0,
            d0_search, fast_opt, t, u);
        TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen, invmap0,
            t, u, simplify_step, score_sum_method, local_d0_search, Lnorm,
            score_d8, d0);
        if (TM>TMmax) TMmax = TM;
        if (TMcut>0) copy_t_u(t, u, t0, u0);
        //run dynamic programing iteratively to find the best alignment
        TM = DP_iter_dimer(r1, r2, xtm, ytm, xt, path, val, xa, ya, xlen, ylen,
             mask, t, u, invmap, 0, 2, (fast_opt)?2:30,
             local_d0_search, D0_MIN, Lnorm, d0, score_d8);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            if (TMcut>0) copy_t_u(t, u, t0, u0);
        }

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.5*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 2;
            }
        }

        /************************************************************/
        /*    get initial alignment based on secondary structure    */
        /************************************************************/
        get_initial_ss_dimer(path, val, secx, secy, xlen, ylen, mask, invmap);
        TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen, invmap,
            t, u, simplify_step, score_sum_method, local_d0_search, Lnorm,
            score_d8, d0);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            if (TMcut>0) copy_t_u(t, u, t0, u0);
        }
        if (TM > TMmax*0.2)
        {
            TM = DP_iter_dimer(r1, r2, xtm, ytm, xt, path, val, xa, ya,
                xlen, ylen, mask, t, u, invmap, 0, 2,
                (fast_opt)?2:30, local_d0_search, D0_MIN, Lnorm, d0, score_d8);
            if (TM>TMmax)
            {
                TMmax = TM;
                for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                if (TMcut>0) copy_t_u(t, u, t0, u0);
            }
        }

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.52*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 3;
            }
        }

        /************************************************************/
        /*    get initial alignment based on local superposition    */
        /************************************************************/
        //=initial5 in original TM-align
        if (get_initial5_dimer( r1, r2, xtm, ytm, path, val, xa, ya,
            xlen, ylen, mask, invmap, d0, d0_search, fast_opt, D0_MIN))
        {
            TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen,
                invmap, t, u, simplify_step, score_sum_method,
                local_d0_search, Lnorm, score_d8, d0);
            if (TM>TMmax)
            {
                TMmax = TM;
                for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                if (TMcut>0) copy_t_u(t, u, t0, u0);
            }
            if (TM > TMmax*ddcc)
            {
                TM = DP_iter_dimer(r1, r2, xtm, ytm, xt, path, val, xa, ya,
                    xlen, ylen, mask, t, u, invmap, 0, 2, 2,
                    local_d0_search, D0_MIN, Lnorm, d0, score_d8);
                if (TM>TMmax)
                {
                    TMmax = TM;
                    for (int i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                    if (TMcut>0) copy_t_u(t, u, t0, u0);
                }
            }
        }
        else
            cerr << "\n\nWarning: initial alignment from local superposition fail!\n\n" << endl;

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.54*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 4;
            }
        }

        /********************************************************************/
        /* get initial alignment by local superposition+secondary structure */
        /********************************************************************/
        //=initial3 in original TM-align
        get_initial_ssplus_dimer(r1, r2, score, path, val, secx, secy, xa, ya,
            xlen, ylen, mask, invmap0, invmap, D0_MIN, d0);
        TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen, invmap,
             t, u, simplify_step, score_sum_method, local_d0_search, Lnorm,
             score_d8, d0);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            if (TMcut>0) copy_t_u(t, u, t0, u0);
        }
        if (TM > TMmax*ddcc)
        {
            TM = DP_iter_dimer(r1, r2, xtm, ytm, xt, path, val, xa, ya,
                xlen, ylen, mask, t, u, invmap, 0, 2,
                (fast_opt)?2:30, local_d0_search, D0_MIN, Lnorm, d0, score_d8);
            if (TM>TMmax)
            {
                TMmax = TM;
                for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                if (TMcut>0) copy_t_u(t, u, t0, u0);
            }
        }

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.56*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 5;
            }
        }

        /*******************************************************************/
        /*    get initial alignment based on fragment gapless threading    */
        /*******************************************************************/
        //=initial4 in original TM-align
        get_initial_fgt(r1, r2, xtm, ytm, xa, ya, xlen, ylen,
            invmap, d0, d0_search, dcu0, fast_opt, t, u);
        TM = detailed_search(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen, invmap,
            t, u, simplify_step, score_sum_method, local_d0_search, Lnorm,
            score_d8, d0);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            if (TMcut>0) copy_t_u(t, u, t0, u0);
        }
        if (TM > TMmax*ddcc)
        {
            TM = DP_iter_dimer(r1, r2, xtm, ytm, xt, path, val, xa, ya,
                xlen, ylen, mask, t, u, invmap, 1, 2, 2,
                local_d0_search, D0_MIN, Lnorm, d0, score_d8);
            if (TM>TMmax)
            {
                TMmax = TM;
                for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
                if (TMcut>0) copy_t_u(t, u, t0, u0);
            }
        }

        if (TMcut>0) // pre-terminate if TM-score is too low
        {
            double TMtmp=approx_TM(xlen, ylen, a_opt,
                xa, ya, t0, u0, invmap0, mol_type);

            if (TMtmp<0.58*TMcut)
            {
                TM1=TM2=TM3=TM4=TM5=TMtmp;
                clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                    xtm, ytm, xt, r1, r2, xlen, minlen);
                return 6;
            }
        }

        //************************************************//
        //    get initial alignment from user's input:    //
        //************************************************//
        if (i_opt==1)// if input has set parameter for "-i"
        {
            for (int j = 0; j < ylen; j++)// Set aligned position to be "-1"
                invmap[j] = -1;

            int i1 = -1;// in C version, index starts from zero, not from one
            int i2 = -1;
            int L1 = sequence[0].size();
            int L2 = sequence[1].size();
            int L = min(L1, L2);// Get positions for aligned residues
            for (int kk1 = 0; kk1 < L; kk1++)
            {
                if (sequence[0][kk1] != '-')
                    i1++;
                if (sequence[1][kk1] != '-')
                {
                    i2++;
                    if (i2 >= ylen || i1 >= xlen) kk1 = L;
                    else if (sequence[0][kk1] != '-') invmap[i2] = i1;
                }
            }

            //--------------- 2. Align proteins from original alignment
            double prevD0_MIN = D0_MIN;// stored for later use
            int prevLnorm = Lnorm;
            double prevd0 = d0;
            TM_ali = standard_TMscore(r1, r2, xtm, ytm, xt, xa, ya,
                xlen, ylen, invmap, L_ali, rmsd_ali, D0_MIN, Lnorm, d0,
                d0_search, score_d8, t, u, mol_type);
            D0_MIN = prevD0_MIN;
            Lnorm = prevLnorm;
            d0 = prevd0;

            TM = detailed_search_standard(r1, r2, xtm, ytm, xt, xa, ya,
                xlen, ylen, invmap, t, u, 40, 8, local_d0_search, true, Lnorm,
                score_d8, d0);
            if (TM > TMmax)
            {
                TMmax = TM;
                for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            }
            // Different from get_initial, get_initial_ss and get_initial_ssplus
            TM = DP_iter_dimer(r1, r2, xtm, ytm, xt, path, val, xa, ya,
                xlen, ylen, mask, t, u, invmap, 0, 2,
                (fast_opt)?2:30, local_d0_search, D0_MIN, Lnorm, d0, score_d8);
            if (TM>TMmax)
            {
                TMmax = TM;
                for (i = 0; i<ylen; i++) invmap0[i] = invmap[i];
            }
        }
    }



    //*******************************************************************//
    //    The alignment will not be changed any more in the following    //
    //*******************************************************************//
    //check if the initial alignment is generated appropriately
    bool flag=false;
    for(i=0; i<ylen; i++)
    {
        if(invmap0[i]>=0)
        {
            flag=true;
            break;
        }
    }
    if(!flag)
    {
        cout << "There is no alignment between the two proteins! "
             << "Program stop with no result!" << endl;
        TM1=TM2=TM3=TM4=TM5=0;
        return 1;
    }

    /* last TM-score pre-termination */
    if (TMcut>0)
    {
        double TMtmp=approx_TM(xlen, ylen, a_opt,
            xa, ya, t0, u0, invmap0, mol_type);

        if (TMtmp<0.6*TMcut)
        {
            TM1=TM2=TM3=TM4=TM5=TMtmp;
            clean_up_after_approx_TM(invmap0, invmap, score, path, val,
                xtm, ytm, xt, r1, r2, xlen, minlen);
            return 7;
        }
    }

    //********************************************************************//
    //    Detailed TMscore search engine --> prepare for final TMscore    //
    //********************************************************************//
    //run detailed TMscore search engine for the best alignment, and
    //extract the best rotation matrix (t, u) for the best alignment
    simplify_step=1;
    if (fast_opt) simplify_step=40;
    score_sum_method=8;
    TM = detailed_search_standard(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen,
        invmap0, t, u, simplify_step, score_sum_method, local_d0_search,
        false, Lnorm, score_d8, d0);

    //select pairs with dis<d8 for final TMscore computation and output alignment
    int k=0;
    int *m1, *m2;
    double d;
    m1=new int[xlen]; //alignd index in x
    m2=new int[ylen]; //alignd index in y
    do_rotation(xa, xt, xlen, t, u);
    k=0;
    for(int j=0; j<ylen; j++)
    {
        i=invmap0[j];
        if(i>=0)//aligned
        {
            n_ali++;
            d=sqrt(dist(&xt[i][0], &ya[j][0]));
            if (d <= score_d8 || (i_opt == 3))
            {
                m1[k]=i;
                m2[k]=j;

                xtm[k][0]=xa[i][0];
                xtm[k][1]=xa[i][1];
                xtm[k][2]=xa[i][2];

                ytm[k][0]=ya[j][0];
                ytm[k][1]=ya[j][1];
                ytm[k][2]=ya[j][2];

                r1[k][0] = xt[i][0];
                r1[k][1] = xt[i][1];
                r1[k][2] = xt[i][2];
                r2[k][0] = ya[j][0];
                r2[k][1] = ya[j][1];
                r2[k][2] = ya[j][2];

                k++;
            }
        }
    }
    n_ali8=k;

    Kabsch(r1, r2, n_ali8, 0, &rmsd0, t, u);// rmsd0 is used for final output, only recalculate rmsd0, not t & u
    rmsd0 = sqrt(rmsd0 / n_ali8);


    //****************************************//
    //              Final TMscore             //
    //    Please set parameters for output    //
    //****************************************//
    double rmsd;
    simplify_step=1;
    score_sum_method=0;
    double Lnorm_0=ylen;


    //normalized by length of structure A
    parameter_set4final(Lnorm_0, D0_MIN, Lnorm, d0, d0_search, mol_type);
    d0A=d0;
    d0_0=d0A;
    local_d0_search = d0_search;
    TM1 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0, simplify_step,
        score_sum_method, &rmsd, local_d0_search, Lnorm, score_d8, d0);
    TM_0 = TM1;

    //normalized by length of structure B
    parameter_set4final(xlen+0.0, D0_MIN, Lnorm, d0, d0_search, mol_type);
    d0B=d0;
    local_d0_search = d0_search;
    TM2 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t, u, simplify_step,
        score_sum_method, &rmsd, local_d0_search, Lnorm, score_d8, d0);

    double Lnorm_d0;
    if (a_opt>0)
    {
        //normalized by average length of structures A, B
        Lnorm_0=(xlen+ylen)*0.5;
        parameter_set4final(Lnorm_0, D0_MIN, Lnorm, d0, d0_search, mol_type);
        d0a=d0;
        d0_0=d0a;
        local_d0_search = d0_search;

        TM3 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0,
            simplify_step, score_sum_method, &rmsd, local_d0_search, Lnorm,
            score_d8, d0);
        TM_0=TM3;
    }
    if (u_opt)
    {
        //normalized by user assigned length
        parameter_set4final(Lnorm_ass, D0_MIN, Lnorm,
            d0, d0_search, mol_type);
        d0u=d0;
        d0_0=d0u;
        Lnorm_0=Lnorm_ass;
        local_d0_search = d0_search;
        TM4 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0,
            simplify_step, score_sum_method, &rmsd, local_d0_search, Lnorm,
            score_d8, d0);
        TM_0=TM4;
    }
    if (d_opt)
    {
        //scaled by user assigned d0
        parameter_set4scale(ylen, d0_scale, Lnorm, d0, d0_search);
        d0_out=d0_scale;
        d0_0=d0_scale;
        //Lnorm_0=ylen;
        Lnorm_d0=Lnorm_0;
        local_d0_search = d0_search;
        TM5 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0,
            simplify_step, score_sum_method, &rmsd, local_d0_search, Lnorm,
            score_d8, d0);
        TM_0=TM5;
    }

    /* derive alignment from superposition */
    int ali_len=xlen+ylen; //maximum length of alignment
    seqxA.assign(ali_len,'-');
    seqM.assign( ali_len,' ');
    seqyA.assign(ali_len,'-');
    
    //do_rotation(xa, xt, xlen, t, u);
    do_rotation(xa, xt, xlen, t0, u0);

    int kk=0, i_old=0, j_old=0;
    d=0;
    for(int k=0; k<n_ali8; k++)
    {
        for(int i=i_old; i<m1[k]; i++)
        {
            //align x to gap
            seqxA[kk]=seqx[i];
            seqyA[kk]='-';
            seqM[kk]=' ';                    
            kk++;
        }

        for(int j=j_old; j<m2[k]; j++)
        {
            //align y to gap
            seqxA[kk]='-';
            seqyA[kk]=seqy[j];
            seqM[kk]=' ';
            kk++;
        }

        seqxA[kk]=seqx[m1[k]];
        seqyA[kk]=seqy[m2[k]];
        Liden+=(seqxA[kk]==seqyA[kk]);
        d=sqrt(dist(&xt[m1[k]][0], &ya[m2[k]][0]));
        if(d<d0_out) seqM[kk]=':';
        else         seqM[kk]='.';
        kk++;  
        i_old=m1[k]+1;
        j_old=m2[k]+1;
    }

    //tail
    for(int i=i_old; i<xlen; i++)
    {
        //align x to gap
        seqxA[kk]=seqx[i];
        seqyA[kk]='-';
        seqM[kk]=' ';
        kk++;
    }    
    for(int j=j_old; j<ylen; j++)
    {
        //align y to gap
        seqxA[kk]='-';
        seqyA[kk]=seqy[j];
        seqM[kk]=' ';
        kk++;
    }
    seqxA=seqxA.substr(0,kk);
    seqyA=seqyA.substr(0,kk);
    seqM =seqM.substr(0,kk);

    /* free memory */
    clean_up_after_approx_TM(invmap0, invmap, score, path, val,
        xtm, ytm, xt, r1, r2, xlen, minlen);
    delete [] m1;
    delete [] m2;
    return 0; // zero for no exception
}

void MMalign_dimer(double & total_score, 
    const vector<vector<vector<double> > >&xa_vec,
    const vector<vector<vector<double> > >&ya_vec,
    const vector<vector<char> >&seqx_vec, const vector<vector<char> >&seqy_vec,
    const vector<vector<char> >&secx_vec, const vector<vector<char> >&secy_vec,
    const vector<int> &mol_vec1, const vector<int> &mol_vec2,
    const vector<int> &xlen_vec, const vector<int> &ylen_vec,
    double **xa, double **ya, char *seqx, char *seqy, char *secx, char *secy,
    int len_aa, int len_na, int chain1_num, int chain2_num, double **TMave_mat,
    vector<vector<string> >&seqxA_mat, vector<vector<string> >&seqyA_mat,
    int *assign1_list, int *assign2_list, vector<string>&sequence,
    double d0_scale, bool fast_opt)
{
    int i,j;
    int xlen=0;
    int ylen=0;
    vector<int> xlen_dimer;
    vector<int> ylen_dimer;
    for (i=0;i<chain1_num;i++)
    {
        j=assign1_list[i];
        if (j<0) continue;
        xlen+=xlen_vec[i];
        ylen+=ylen_vec[j];
        xlen_dimer.push_back(xlen_vec[i]);
        ylen_dimer.push_back(ylen_vec[j]);
    }
    if (xlen<=3 || ylen<=3) return;

    bool **mask; // mask out inter-chain region
    NewArray(&mask, xlen+1, ylen+1);
    for (i=0;i<xlen+1;i++) for (j=0;j<ylen+1;j++) mask[i][j]=false;
    for (i=0;i<xlen_dimer[0]+1;i++) mask[i][0]=true;
    for (j=0;j<ylen_dimer[0]+1;j++) mask[0][j]=true;
    int c,prev_xlen,prev_ylen;
    prev_xlen=1;
    prev_ylen=1;
    for (c=0;c<xlen_dimer.size();c++)
    {
        for (i=prev_xlen;i<prev_xlen+xlen_dimer[c];i++)
            for (j=prev_ylen;j<prev_ylen+ylen_dimer[c];j++) mask[i][j]=true;
        prev_xlen+=xlen_dimer[c];
        prev_ylen+=ylen_dimer[c];
    }
    vector<int>().swap(xlen_dimer);
    vector<int>().swap(ylen_dimer);

    seqx = new char[xlen+1];
    secx = new char[xlen+1];
    NewArray(&xa, xlen, 3);
    seqy = new char[ylen+1];
    secy = new char[ylen+1];
    NewArray(&ya, ylen, 3);

    int mol_type=copy_chain_pair_data(xa_vec, ya_vec, seqx_vec, seqy_vec,
        secx_vec, secy_vec, mol_vec1, mol_vec2, xlen_vec, ylen_vec,
        xa, ya, seqx, seqy, secx, secy, chain1_num, chain2_num,
        seqxA_mat, seqyA_mat, assign1_list, assign2_list, sequence);

    /* declare variable specific to this pair of TMalign */
    double t0[3], u0[3][3];
    double TM1, TM2;
    double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
    double d0_0, TM_0;
    double d0A, d0B, d0u, d0a;
    double d0_out=5.0;
    string seqM, seqxA, seqyA;// for output alignment
    double rmsd0 = 0.0;
    int L_ali;                // Aligned length in standard_TMscore
    double Liden=0;
    double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
    int n_ali=0;
    int n_ali8=0;

    double Lnorm_ass=len_aa+len_na;

    TMalign_dimer_main(xa, ya, seqx, seqy, secx, secy,
        t0, u0, TM1, TM2, TM3, TM4, TM5,
        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
        xlen, ylen, mask, sequence, Lnorm_ass, d0_scale,
        1, false, true, false, fast_opt, mol_type, -1);

    /* clean up TM-align */
    delete [] seqx;
    delete [] seqy;
    delete [] secx;
    delete [] secy;
    DeleteArray(&xa,xlen);
    DeleteArray(&ya,ylen);
    DeleteArray(&mask,xlen+1);

    /* re-compute chain level alignment */
    total_score=0;
    for (i=0;i<chain1_num;i++)
    {
        xlen=xlen_vec[i];
        if (xlen<3)
        {
            for (j=0;j<chain2_num;j++) TMave_mat[i][j]=-1;
            continue;
        }
        seqx = new char[xlen+1];
        secx = new char[xlen+1];
        NewArray(&xa, xlen, 3);
        copy_chain_data(xa_vec[i],seqx_vec[i],secx_vec[i],
            xlen,xa,seqx,secx);

        double **xt;
        NewArray(&xt, xlen, 3);
        do_rotation(xa, xt, xlen, t0, u0);

        for (j=0;j<chain2_num;j++)
        {
            if (mol_vec1[i]*mol_vec2[j]<0) //no protein-RNA alignment
            {
                TMave_mat[i][j]=-1;
                continue;
            }

            ylen=ylen_vec[j];
            if (ylen<3)
            {
                TMave_mat[i][j]=-1;
                continue;
            }
            seqy = new char[ylen+1];
            secy = new char[ylen+1];
            NewArray(&ya, ylen, 3);
            copy_chain_data(ya_vec[j],seqy_vec[j],secy_vec[j],
                ylen,ya,seqy,secy);

            /* declare variable specific to this pair of TMalign */
            d0_out=5.0;
            seqM.clear();
            seqxA.clear();
            seqyA.clear();
            rmsd0 = 0.0;
            Liden=0;
            int *invmap = new int[ylen+1];

            double Lnorm_ass=len_aa;
            if (mol_vec1[i]+mol_vec2[j]>0) Lnorm_ass=len_na;

            /* entry function for structure alignment */
            se_main(xt, ya, seqx, seqy, TM1, TM2, TM3, TM4, TM5,
                d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
                rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                xlen, ylen, sequence, Lnorm_ass, d0_scale,
                0, false, 2, false, mol_vec1[i]+mol_vec2[j], 1, invmap);

            /* print result */
            seqxA_mat[i][j]=seqxA;
            seqyA_mat[i][j]=seqyA;

            TMave_mat[i][j]=TM4*Lnorm_ass;
            if (assign1_list[i]==j)
            {
                if (TM4<=0) assign1_list[i]=assign2_list[j]=-1;
                else        total_score+=TMave_mat[i][j];
            }

            /* clean up */
            seqM.clear();
            seqxA.clear();
            seqyA.clear();

            delete[]seqy;
            delete[]secy;
            DeleteArray(&ya,ylen);
            delete[]invmap;
        }
        delete[]seqx;
        delete[]secx;
        DeleteArray(&xa,xlen);
        DeleteArray(&xt,xlen);
    }
    return;
}

/* return the number of chains that are trimmed */
int trimComplex(vector<vector<vector<double> > >&a_trim_vec,
    vector<vector<char> >&seq_trim_vec, vector<vector<char> >&sec_trim_vec,
    vector<int>&len_trim_vec,
    const vector<vector<vector<double> > >&a_vec,
    const vector<vector<char> >&seq_vec, const vector<vector<char> >&sec_vec,
    const vector<int> &len_vec, const vector<int> &mol_vec,
    const int Lchain_aa_max, const int Lchain_na_max)
{
    int trim_chain_count=0;
    int chain_num=a_vec.size();
    int i,j;
    int r1,r2;
    double dinter;
    double dinter_min;
    vector<pair<double,int> >dinter_vec;
    vector<bool> include_vec;
    vector<char> seq_empty;
    vector<vector<double> >  a_empty;
    vector<double> xcoor(3,0);
    vector<double> ycoor(3,0);
    int xlen,ylen;
    int Lchain_max;
    double expand=2;
    for (i=0;i<chain_num;i++)
    {
        xlen=len_vec[i];
        if (mol_vec[i]>0) Lchain_max=Lchain_na_max*expand;
        else              Lchain_max=Lchain_aa_max*expand;
        if (Lchain_max<3) Lchain_max=3;
        if (xlen<=Lchain_max || xlen<=3)
        {
            a_trim_vec.push_back(a_vec[i]);
            seq_trim_vec.push_back(seq_vec[i]);
            sec_trim_vec.push_back(sec_vec[i]);
            len_trim_vec.push_back(xlen);
            continue;
        }
        trim_chain_count++;
        for (r1=0;r1<xlen;r1++)
        {
            xcoor[0]=a_vec[i][r1][0];
            xcoor[1]=a_vec[i][r1][1];
            xcoor[2]=a_vec[i][r1][2];
            dinter_min=FLT_MAX;
            for (j=0;j<chain_num;j++)
            {
                if (i==j) continue;
                ylen=len_vec[j];
                for (r2=0;r2<ylen;r2++)
                {
                    ycoor[0]=a_vec[j][r2][0];
                    ycoor[1]=a_vec[j][r2][1];
                    ycoor[2]=a_vec[j][r2][2];
                    dinter=(xcoor[0]-ycoor[0])*(xcoor[0]-ycoor[0])+
                           (xcoor[1]-ycoor[1])*(xcoor[1]-ycoor[1])+
                           (xcoor[2]-ycoor[2])*(xcoor[2]-ycoor[2]);
                    if (dinter<dinter_min) dinter_min=dinter;
                }
            }
            dinter_vec.push_back(make_pair(dinter,r1));
        }
        sort(dinter_vec.begin(),dinter_vec.end());
        include_vec.assign(xlen,false);
        for (r1=0;r1<Lchain_max;r1++)
            include_vec[dinter_vec[r1].second]=true;
        dinter_vec.clear();

        a_trim_vec.push_back(a_empty);
        seq_trim_vec.push_back(seq_empty);
        sec_trim_vec.push_back(seq_empty);
        len_trim_vec.push_back(Lchain_max);
        for (r1=0;r1<xlen;r1++)
        {
            if (include_vec[r1]==false) continue;
            a_trim_vec[i].push_back(a_vec[i][r1]);
            seq_trim_vec[i].push_back(seq_vec[i][r1]);
            sec_trim_vec[i].push_back(sec_vec[i][r1]);
        }
        include_vec.clear();
    }
    vector<pair<double,int> >().swap(dinter_vec);
    vector<bool>().swap(include_vec);
    vector<double> ().swap(xcoor);
    vector<double> ().swap(ycoor);
    return trim_chain_count;
}

void output_dock_rotation_matrix(const char* fname_matrix,
    const vector<string>&xname_vec, const vector<string>&yname_vec,
    double ** ut_mat, int *assign1_list)
{
    fstream fout;
    fout.open(fname_matrix, ios::out | ios::trunc);
    if (fout)// succeed
    {
        int i,k;
        for (i=0;i<xname_vec.size();i++)
        {
            if (assign1_list[i]<0) continue;
            fout << "------ The rotation matrix to rotate "
                 <<xname_vec[i]<<" to "<<yname_vec[i]<<" ------\n"
                 << "m               t[m]        u[m][0]        u[m][1]        u[m][2]\n";
            for (k = 0; k < 3; k++)
                fout<<k<<setiosflags(ios::fixed)<<setprecision(10)
                    <<' '<<setw(18)<<ut_mat[i][9+k]
                    <<' '<<setw(14)<<ut_mat[i][3*k+0]
                    <<' '<<setw(14)<<ut_mat[i][3*k+1]
                    <<' '<<setw(14)<<ut_mat[i][3*k+2]<<'\n';
        }
        fout << "\nCode for rotating Structure 1 from (x,y,z) to (X,Y,Z):\n"
                "for(i=0; i<L; i++)\n"
                "{\n"
                "   X[i] = t[0] + u[0][0]*x[i] + u[0][1]*y[i] + u[0][2]*z[i];\n"
                "   Y[i] = t[1] + u[1][0]*x[i] + u[1][1]*y[i] + u[1][2]*z[i];\n"
                "   Z[i] = t[2] + u[2][0]*x[i] + u[2][1]*y[i] + u[2][2]*z[i];\n"
                "}"<<endl;
        fout.close();
    }
    else
        cout << "Open file to output rotation matrix fail.\n";
}

/* SOIalign.h */
/* Non-sequential alignment */

void assign_sec_bond(int **secx_bond, const char *secx, const int xlen)
{
    int i,j;
    int starti=-1;
    int endi=-1;
    char ss;
    char prev_ss=0;
    for (i=0; i<xlen; i++)
    {
        ss=secx[i];
        secx_bond[i][0]=secx_bond[i][1]=-1;
        if (ss!=prev_ss && !(ss=='C' && prev_ss=='T') 
                        && !(ss=='T' && prev_ss=='C'))
        {
            if (starti>=0) // previous SSE end
            {
                endi=i;
                for (j=starti;j<endi;j++)
                {
                    secx_bond[j][0]=starti;
                    secx_bond[j][1]=endi;
                }
            }
            if (ss=='H' || ss=='E' || ss=='<' || ss=='>') starti=i;
            else starti=-1;
        }
        prev_ss=secx[i];
    }
    if (starti>=0) // previous SSE end
    {
        endi=i;
        for (j=starti;j<endi;j++)
        {
            secx_bond[j][0]=starti;
            secx_bond[j][1]=endi;
        }
    }
    for (i=0;i<xlen;i++) if (secx_bond[i][1]-secx_bond[i][0]==1)
        secx_bond[i][0]=secx_bond[i][1]=-1;
}

void getCloseK(double **xa, const int xlen, const int closeK_opt, double **xk)
{
    double **score;
    NewArray(&score, xlen+1, xlen+1);
    vector<pair<double,int> > close_idx_vec(xlen, make_pair(0,0));
    int i,j,k;
    for (i=0;i<xlen;i++)
    {
        score[i+1][i+1]=0;
        for (j=i+1;j<xlen;j++) score[j+1][i+1]=score[i+1][j+1]=dist(xa[i], xa[j]);
    }
    for (i=0;i<xlen;i++)
    {
        for (j=0;j<xlen;j++)
        {
            close_idx_vec[j].first=score[i+1][j+1];
            close_idx_vec[j].second=j;
        }
        sort(close_idx_vec.begin(), close_idx_vec.end());
        for (k=0;k<closeK_opt;k++)
        {
            j=close_idx_vec[k % xlen].second;
            xk[i*closeK_opt+k][0]=xa[j][0];
            xk[i*closeK_opt+k][1]=xa[j][1];
            xk[i*closeK_opt+k][2]=xa[j][2];
        }
    }

    /* clean up */
    vector<pair<double,int> >().swap(close_idx_vec);
    DeleteArray(&score, xlen+1);
}

/* check if pairing i to j conform to sequantiality within the SSE */
inline bool sec2sq(const int i, const int j,
    int **secx_bond, int **secy_bond, int *fwdmap, int *invmap)
{
    if (i<0 || j<0) return true;
    int ii,jj;
    if (secx_bond[i][0]>=0)
    {
        for (ii=secx_bond[i][0];ii<secx_bond[i][1];ii++)
        {
            jj=fwdmap[ii];
            if (jj>=0 && (i-ii)*(j-jj)<=0) return false;
        }
    }
    if (secy_bond[j][0]>=0)
    {
        for (jj=secy_bond[j][0];jj<secy_bond[j][1];jj++)
        {
            ii=invmap[jj];
            if (ii>=0 && (i-ii)*(j-jj)<=0) return false;
        }
    }
    return true;
}

void soi_egs(double **score, const int xlen, const int ylen, int *invmap,
    int **secx_bond, int **secy_bond, const int mm_opt)
{
    int i,j;
    int *fwdmap=new int[xlen]; // j=fwdmap[i];
    for (i=0; i<xlen; i++) fwdmap[i]=-1;
    for (j=0; j<ylen; j++)
    {
        i=invmap[j];
        if (i>=0) fwdmap[i]=j;
    }

    /* stage 1 - make initial assignment, starting from the highest score pair */
    double max_score;
    int maxi,maxj;
    while(1)
    {
        max_score=0;
        maxi=maxj=-1;
        for (i=0;i<xlen;i++)
        {
            if (fwdmap[i]>=0) continue;
            for (j=0;j<ylen;j++)
            {
                if (invmap[j]>=0 || score[i+1][j+1]<=max_score) continue;
                if (mm_opt==6 && !sec2sq(i,j,secx_bond,secy_bond,
                    fwdmap,invmap)) continue;
                maxi=i;
                maxj=j;
                max_score=score[i+1][j+1];
            }
        }
        if (maxi<0) break; // no assignment;
        invmap[maxj]=maxi;
        fwdmap[maxi]=maxj;
    }

    double total_score=0;
    for (j=0;j<ylen;j++)
    {
        i=invmap[j];
        if (i>=0) total_score+=score[i+1][j+1];
    }

    /* stage 2 - swap assignment until total score cannot be improved */
    int iter;
    int oldi,oldj;
    double delta_score;
    for (iter=0; iter<getmin(xlen,ylen)*5; iter++)
    {
        //cout<<"total_score="<<total_score<<".iter="<<iter<<endl;
        //print_invmap(invmap,ylen);
        delta_score=-1;
        for (i=0;i<xlen;i++)
        {
            oldj=fwdmap[i];
            for (j=0;j<ylen;j++)
            {
                oldi=invmap[j];
                if (score[i+1][j+1]<=0 || oldi==i) continue;
                if (mm_opt==6 && (!sec2sq(i,j,secx_bond,secy_bond,fwdmap,invmap) ||
                            !sec2sq(oldi,oldj,secx_bond,secy_bond,fwdmap,invmap)))
                    continue;
                delta_score=score[i+1][j+1];
                if (oldi>=0 && oldj>=0) delta_score+=score[oldi+1][oldj+1];
                if (oldi>=0) delta_score-=score[oldi+1][j+1];
                if (oldj>=0) delta_score-=score[i+1][oldj+1];

                if (delta_score>0) // successful swap
                {
                    fwdmap[i]=j;
                    if (oldi>=0) fwdmap[oldi]=oldj;
                    invmap[j]=i;
                    if (oldj>=0) invmap[oldj]=oldi;
                    total_score+=delta_score;
                    break;
                }
            }
        }
        if (delta_score<=0) break; // cannot make further swap
    }

    /* clean up */
    delete[]fwdmap;
}

/* entry function for se
 * u_opt corresponds to option -L
 *       if u_opt==2, use d0 from Lnorm_ass for alignment
 * */
int soi_se_main(
    double **xa, double **ya, const char *seqx, const char *seqy,
    double &TM1, double &TM2, double &TM3, double &TM4, double &TM5,
    double &d0_0, double &TM_0,
    double &d0A, double &d0B, double &d0u, double &d0a, double &d0_out,
    string &seqM, string &seqxA, string &seqyA,
    double &rmsd0, int &L_ali, double &Liden,
    double &TM_ali, double &rmsd_ali, int &n_ali, int &n_ali8,
    const int xlen, const int ylen, 
    const double Lnorm_ass, const double d0_scale, const bool i_opt,
    const bool a_opt, const int u_opt, const bool d_opt, const int mol_type,
    const int outfmt_opt, int *invmap, double *dist_list,
    int **secx_bond, int **secy_bond, const int mm_opt)
{
    double D0_MIN;        //for d0
    double Lnorm;         //normalization length
    double score_d8,d0,d0_search,dcu0;//for TMscore search
    double **score;       // score for aligning a residue pair
    bool   **path;        // for dynamic programming  
    double **val;         // for dynamic programming  

    int *m1=NULL;
    int *m2=NULL;
    int i,j;
    double d;
    if (outfmt_opt<2)
    {
        m1=new int[xlen]; //alignd index in x
        m2=new int[ylen]; //alignd index in y
    }

    /***********************/
    /* allocate memory     */
    /***********************/
    NewArray(&score, xlen+1, ylen+1);
    NewArray(&path,  xlen+1, ylen+1);
    NewArray(&val,   xlen+1, ylen+1);
    //int *invmap          = new int[ylen+1];

    /* set d0 */
    parameter_set4search(xlen, ylen, D0_MIN, Lnorm,
        score_d8, d0, d0_search, dcu0); // set score_d8
    parameter_set4final(xlen, D0_MIN, Lnorm,
        d0B, d0_search, mol_type); // set d0B
    parameter_set4final(ylen, D0_MIN, Lnorm,
        d0A, d0_search, mol_type); // set d0A
    if (a_opt)
        parameter_set4final((xlen+ylen)*0.5, D0_MIN, Lnorm,
            d0a, d0_search, mol_type); // set d0a
    if (u_opt)
    {
        parameter_set4final(Lnorm_ass, D0_MIN, Lnorm,
            d0u, d0_search, mol_type); // set d0u
        if (u_opt==2)
        {
            parameter_set4search(Lnorm_ass, Lnorm_ass, D0_MIN, Lnorm,
                score_d8, d0, d0_search, dcu0); // set score_d8
        }
    }

    /* perform alignment */
    for(j=0; j<ylen; j++) invmap[j]=-1;
    double d02=d0*d0;
    double score_d82=score_d8*score_d8;
    double d2;
    for(i=0; i<xlen; i++)
    {
        for(j=0; j<ylen; j++)
        {
            d2=dist(xa[i], ya[j]);
            if (d2>score_d82) score[i+1][j+1]=0;
            else score[i+1][j+1]=1./(1+ d2/d02);
        }
    }
    if (mm_opt==6) NWDP_TM(score, path, val, xlen, ylen, -0.6, invmap);
    soi_egs(score, xlen, ylen, invmap, secx_bond, secy_bond, mm_opt);

    rmsd0=TM1=TM2=TM3=TM4=TM5=0;
    int k=0;
    n_ali=0;
    n_ali8=0;
    for(j=0; j<ylen; j++)
    {
        i=invmap[j];
        dist_list[j]=-1;
        if(i>=0)//aligned
        {
            n_ali++;
            d=sqrt(dist(&xa[i][0], &ya[j][0]));
            dist_list[j]=d;
            if (score[i+1][j+1]>0)
            {
                if (outfmt_opt<2)
                {
                    m1[k]=i;
                    m2[k]=j;
                }
                k++;
                TM2+=1/(1+(d/d0B)*(d/d0B)); // chain_1
                TM1+=1/(1+(d/d0A)*(d/d0A)); // chain_2
                if (a_opt) TM3+=1/(1+(d/d0a)*(d/d0a)); // -a
                if (u_opt) TM4+=1/(1+(d/d0u)*(d/d0u)); // -u
                if (d_opt) TM5+=1/(1+(d/d0_scale)*(d/d0_scale)); // -d
                rmsd0+=d*d;
            }
        }
    }
    n_ali8=k;
    TM2/=xlen;
    TM1/=ylen;
    TM3/=(xlen+ylen)*0.5;
    TM4/=Lnorm_ass;
    TM5/=ylen;
    if (n_ali8) rmsd0=sqrt(rmsd0/n_ali8);

    if (outfmt_opt>=2)
    {
        DeleteArray(&score, xlen+1);
        return 0;
    }

    /* extract aligned sequence */
    int ali_len=xlen+ylen;
    for (j=0;j<ylen;j++) ali_len-=(invmap[j]>=0);
    seqxA.assign(ali_len,'-');
    seqM.assign( ali_len,' ');
    seqyA.assign(ali_len,'-');

    int *fwdmap = new int [xlen+1];
    for (i=0;i<xlen;i++) fwdmap[i]=-1;
    
    for (j=0;j<ylen;j++)
    {
        seqyA[j]=seqy[j];
        i=invmap[j];
        if (i<0) continue;
        if (sqrt(dist(xa[i], ya[j]))<d0_out) seqM[j]=':';
        else seqM[j]='.';
        fwdmap[i]=j;
        seqxA[j]=seqx[i];
        Liden+=(seqxA[k]==seqyA[k]);
    }
    k=0;
    for (i=0;i<xlen;i++)
    {
        j=fwdmap[i];
        if (j>=0) continue;
        seqxA[ylen+k]=seqx[i];
        k++;
    }

    /* free memory */
    delete [] fwdmap;
    delete [] m1;
    delete [] m2;
    DeleteArray(&score, xlen+1);
    DeleteArray(&path, xlen+1);
    DeleteArray(&val, xlen+1);
    return 0; // zero for no exception
}

inline void SOI_super2score(double **xt, double **ya, const int xlen,
    const int ylen, double **score, double d0, double score_d8)
{
    int i,j;
    double d02=d0*d0;
    double score_d82=score_d8*score_d8;
    double d2;
    for (i=0; i<xlen; i++)
    {
        for(j=0; j<ylen; j++)
        {
            d2=dist(xt[i], ya[j]);
            if (d2>score_d82) score[i+1][j+1]=0;
            else score[i+1][j+1]=1./(1+ d2/d02);
        }
    }
}

//heuristic run of dynamic programing iteratively to find the best alignment
//input: initial rotation matrix t, u
//       vectors x and y, d0
//output: best alignment that maximizes the TMscore, will be stored in invmap
double SOI_iter(double **r1, double **r2, double **xtm, double **ytm,
    double **xt, double **score, bool **path, double **val, double **xa, double **ya,
    int xlen, int ylen, double t[3], double u[3][3], int *invmap0,
    int iteration_max, double local_d0_search,
    double Lnorm, double d0, double score_d8,
    int **secx_bond, int **secy_bond, const int mm_opt, const bool init_invmap=false)
{
    double rmsd; 
    int *invmap=new int[ylen+1];
    
    int iteration, i, j, k;
    double tmscore, tmscore_max, tmscore_old=0;    
    tmscore_max=-1;

    //double d01=d0+1.5;
    double d02=d0*d0;
    double score_d82=score_d8*score_d8;
    double d2;
    for (iteration=0; iteration<iteration_max; iteration++)
    {
        if (iteration==0 && init_invmap) 
            for (j=0;j<ylen;j++) invmap[j]=invmap0[j];
        else
        {
            for (j=0; j<ylen; j++) invmap[j]=-1;
            if (mm_opt==6) NWDP_TM(score, path, val, xlen, ylen, -0.6, invmap);
        }
        soi_egs(score, xlen, ylen, invmap, secx_bond, secy_bond, mm_opt);
    
        k=0;
        for (j=0; j<ylen; j++) 
        {
            i=invmap[j];
            if (i<0) continue;

            xtm[k][0]=xa[i][0];
            xtm[k][1]=xa[i][1];
            xtm[k][2]=xa[i][2];
            
            ytm[k][0]=ya[j][0];
            ytm[k][1]=ya[j][1];
            ytm[k][2]=ya[j][2];
            k++;
        }

        tmscore = TMscore8_search(r1, r2, xtm, ytm, xt, k, t, u,
            40, 8, &rmsd, local_d0_search, Lnorm, score_d8, d0);

        if (tmscore>tmscore_max)
        {
            tmscore_max=tmscore;
            for (j=0; j<ylen; j++) invmap0[j]=invmap[j];
        }
    
        if (iteration>0 && fabs(tmscore_old-tmscore)<0.000001) break;       
        tmscore_old=tmscore;
        do_rotation(xa, xt, xlen, t, u);
        SOI_super2score(xt, ya, xlen, ylen, score, d0, score_d8);
    }// for iteration
    
    delete []invmap;
    return tmscore_max;
}

void get_SOI_initial_assign(double **xk, double **yk, const int closeK_opt,
    double **score, bool **path, double **val, const int xlen, const int ylen,
    double t[3], double u[3][3], int invmap[], 
    double local_d0_search, double d0, double score_d8,
    int **secx_bond, int **secy_bond, const int mm_opt)
{
    int i,j,k;
    double **xfrag;
    double **xtran;
    double **yfrag;
    NewArray(&xfrag, closeK_opt, 3);
    NewArray(&xtran, closeK_opt, 3);
    NewArray(&yfrag, closeK_opt, 3);
    double rmsd;
    double d02=d0*d0;
    double score_d82=score_d8*score_d8;
    double d2;

    /* fill in score */
    for (i=0;i<xlen;i++)
    {
        for (k=0;k<closeK_opt;k++)
        {
            xfrag[k][0]=xk[i*closeK_opt+k][0];
            xfrag[k][1]=xk[i*closeK_opt+k][1];
            xfrag[k][2]=xk[i*closeK_opt+k][2];
        }

        for (j=0;j<ylen;j++)
        {
            for (k=0;k<closeK_opt;k++)
            {
                yfrag[k][0]=yk[j*closeK_opt+k][0];
                yfrag[k][1]=yk[j*closeK_opt+k][1];
                yfrag[k][2]=yk[j*closeK_opt+k][2];
            }
            Kabsch(xfrag, yfrag, closeK_opt, 1, &rmsd, t, u);
            do_rotation(xfrag, xtran, closeK_opt, t, u);
            
            //for (k=0; k<closeK_opt; k++)
            //{
                //d2=dist(xtran[k], yfrag[k]);
                //if (d2>score_d82) score[i+1][j+1]=0;
                //else score[i+1][j+1]=1./(1+d2/d02);
            //}
            k=closeK_opt-1;
            d2=dist(xtran[k], yfrag[k]);
            if (d2>score_d82) score[i+1][j+1]=0;
            else score[i+1][j+1]=1./(1+d2/d02);
        }
    }

    /* initial assignment */
    for (j=0;j<ylen;j++) invmap[j]=-1;
    if (mm_opt==6) NWDP_TM(score, path, val, xlen, ylen, -0.6, invmap);
    for (j=0; j<ylen;j++) i=invmap[j];
    soi_egs(score, xlen, ylen, invmap, secx_bond, secy_bond, mm_opt);

    /* clean up */
    DeleteArray(&xfrag, closeK_opt);
    DeleteArray(&xtran, closeK_opt);
    DeleteArray(&yfrag, closeK_opt);
}

void SOI_assign2super(double **r1, double **r2, double **xtm, double **ytm,
    double **xt, double **xa, double **ya,
    const int xlen, const int ylen, double t[3], double u[3][3], int invmap[], 
    double local_d0_search, double Lnorm, double d0, double score_d8)
{
    int i,j,k;
    double rmsd;
    double d02=d0*d0;
    double score_d82=score_d8*score_d8;
    double d2;

    k=0;
    for (j=0; j<ylen; j++)
    {
        i=invmap[j];
        if (i<0) continue;
        xtm[k][0]=xa[i][0];
        xtm[k][1]=xa[i][1];
        xtm[k][2]=xa[i][2];

        ytm[k][0]=ya[j][0];
        ytm[k][1]=ya[j][1];
        ytm[k][2]=ya[j][2];
        k++;
    }
    TMscore8_search(r1, r2, xtm, ytm, xt, k, t, u,
        40, 8, &rmsd, local_d0_search, Lnorm, score_d8, d0);
    do_rotation(xa, xt, xlen, t, u);
}

/* entry function for TM-align with circular permutation
 * i_opt, a_opt, u_opt, d_opt, TMcut are not implemented yet */
int SOIalign_main(double **xa, double **ya,
    double **xk, double **yk, const int closeK_opt,
    const char *seqx, const char *seqy, const char *secx, const char *secy,
    double t0[3], double u0[3][3],
    double &TM1, double &TM2, double &TM3, double &TM4, double &TM5,
    double &d0_0, double &TM_0,
    double &d0A, double &d0B, double &d0u, double &d0a, double &d0_out,
    string &seqM, string &seqxA, string &seqyA, int *invmap,
    double &rmsd0, int &L_ali, double &Liden,
    double &TM_ali, double &rmsd_ali, int &n_ali, int &n_ali8,
    const int xlen, const int ylen,
    const vector<string> sequence, const double Lnorm_ass,
    const double d0_scale, const int i_opt, const int a_opt,
    const bool u_opt, const bool d_opt, const bool fast_opt,
    const int mol_type, double *dist_list, 
    int **secx_bond, int **secy_bond, const int mm_opt)
{
    double D0_MIN;        //for d0
    double Lnorm;         //normalization length
    double score_d8,d0,d0_search,dcu0;//for TMscore search
    double t[3], u[3][3]; //Kabsch translation vector and rotation matrix
    double **score;       // Input score table for enhanced greedy search
    double **scoret;      // Transposed score table for enhanced greedy search
    bool   **path;        // for dynamic programming  
    double **val;         // for dynamic programming  
    double **xtm, **ytm;  // for TMscore search engine
    double **xt;          //for saving the superposed version of r_1 or xtm
    double **yt;          //for saving the superposed version of r_2 or ytm
    double **r1, **r2;    // for Kabsch rotation

    /***********************/
    /* allocate memory     */
    /***********************/
    int minlen = min(xlen, ylen);
    int maxlen = (xlen>ylen)?xlen:ylen;
    NewArray(&score,  xlen+1, ylen+1);
    NewArray(&scoret, ylen+1, xlen+1);
    NewArray(&path, maxlen+1, maxlen+1);
    NewArray(&val,  maxlen+1, maxlen+1);
    NewArray(&xtm, minlen, 3);
    NewArray(&ytm, minlen, 3);
    NewArray(&xt, xlen, 3);
    NewArray(&yt, ylen, 3);
    NewArray(&r1, minlen, 3);
    NewArray(&r2, minlen, 3);

    /***********************/
    /*    parameter set    */
    /***********************/
    parameter_set4search(xlen, ylen, D0_MIN, Lnorm, 
        score_d8, d0, d0_search, dcu0);
    int simplify_step    = 40; //for simplified search engine
    int score_sum_method = 8;  //for scoring method, whether only sum over pairs with dis<score_d8

    int i,j;
    int *fwdmap0         = new int[xlen+1];
    int *invmap0         = new int[ylen+1];
    
    double TMmax=-1, TM=-1;
    for(i=0; i<xlen; i++) fwdmap0[i]=-1;
    for(j=0; j<ylen; j++) invmap0[j]=-1;
    double local_d0_search = d0_search;
    int iteration_max=(fast_opt)?2:30;
    //if (mm_opt==6) iteration_max=1;

    /*************************************************************/
    /* initial alignment with sequence order dependent alignment */
    /*************************************************************/
    CPalign_main(
        xa, ya, seqx, seqy, secx, secy,
        t0, u0, TM1, TM2, TM3, TM4, TM5,
        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
        xlen, ylen, sequence, Lnorm_ass, d0_scale,
        i_opt, a_opt, u_opt, d_opt, fast_opt,
        mol_type,-1);
    if (mm_opt==6)
    {
        i=0;
        j=0;
        for (int r=0;r<seqxA.size();r++)
        {
            if (seqxA[r]=='*') // circular permutation point
            {
                for (int jj=0;jj<j;jj++) if (invmap0[jj]>=0)
                    invmap0[jj]+=xlen - i;
                i=0;
                continue;
            }
            if (seqyA[r]!='-')
            {
                if (seqxA[r]!='-') invmap0[j]=i;
                j++;
            }
            if (seqxA[r]!='-') i++;
        }
        for (j=0;j<ylen;j++)
        {
            i=invmap0[j];
            if (i>=0) fwdmap0[i]=j;
        }
    }
    do_rotation(xa, xt, xlen, t0, u0);
    SOI_super2score(xt, ya, xlen, ylen, score, d0, score_d8);
    for (i=0;i<xlen;i++) for (j=0;j<ylen;j++) scoret[j+1][i+1]=score[i+1][j+1];
    TMmax=SOI_iter(r1, r2, xtm, ytm, xt, score, path, val, xa, ya,
        xlen, ylen, t0, u0, invmap0, iteration_max,
        local_d0_search, Lnorm, d0, score_d8, secx_bond, secy_bond, mm_opt, true);
    TM   =SOI_iter(r2, r1, ytm, xtm, yt,scoret, path, val, ya, xa,
        ylen, xlen, t0, u0, fwdmap0, iteration_max,
        local_d0_search, Lnorm, d0, score_d8, secy_bond, secx_bond, mm_opt, true);
    //cout<<"TM2="<<TM2<<"\tTM1="<<TM1<<"\tTMmax="<<TMmax<<"\tTM="<<TM<<endl;
    if (TM>TMmax)
    {
        TMmax = TM;
        for (j=0; j<ylen; j++) invmap0[j]=-1;
        for (i=0; i<xlen; i++) 
        {
            j=fwdmap0[i];
            if (j>=0) invmap0[j]=i;
        }
    }
    
    /***************************************************************/
    /* initial alignment with sequence order independent alignment */
    /***************************************************************/
    if (closeK_opt>=3)
    {
        get_SOI_initial_assign(xk, yk, closeK_opt, score, path, val,
            xlen, ylen, t, u, invmap, local_d0_search, d0, score_d8,
            secx_bond, secy_bond, mm_opt);
        for (i=0;i<xlen;i++) for (j=0;j<ylen;j++) scoret[j+1][i+1]=score[i+1][j+1];

        SOI_assign2super(r1, r2, xtm, ytm, xt, xa, ya,
            xlen, ylen, t, u, invmap, local_d0_search, Lnorm, d0, score_d8);
        TM=SOI_iter(r1, r2, xtm, ytm, xt, score, path, val, xa, ya,
            xlen, ylen, t, u, invmap, iteration_max,
            local_d0_search, Lnorm, d0, score_d8, secx_bond, secy_bond, mm_opt);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (j = 0; j<ylen; j++) invmap0[j] = invmap[j];
        }

        for (i=0;i<xlen;i++) fwdmap0[i]=-1;
        if (mm_opt==6) NWDP_TM(scoret, path, val, ylen, xlen, -0.6, fwdmap0);
        soi_egs(scoret, ylen, xlen, fwdmap0, secy_bond, secx_bond, mm_opt);
        SOI_assign2super(r2, r1, ytm, xtm, yt, ya, xa,
            ylen, xlen, t, u, fwdmap0, local_d0_search, Lnorm, d0, score_d8);
        TM=SOI_iter(r2, r1, ytm, xtm, yt, scoret, path, val, ya, xa, ylen, xlen, t, u,
            fwdmap0, iteration_max, local_d0_search, Lnorm, d0, score_d8,secy_bond, secx_bond, mm_opt);
        if (TM>TMmax)
        {
            TMmax = TM;
            for (j=0; j<ylen; j++) invmap0[j]=-1;
            for (i=0; i<xlen; i++) 
            {
                j=fwdmap0[i];
                if (j>=0) invmap0[j]=i;
            }
        }
    }

    //*******************************************************************//
    //    The alignment will not be changed any more in the following    //
    //*******************************************************************//
    //check if the initial alignment is generated appropriately
    bool flag=false;
    for (i=0; i<xlen; i++) fwdmap0[i]=-1;
    for (j=0; j<ylen; j++)
    {
        i=invmap0[j];
        invmap[j]=i;
        if (i>=0)
        {
            fwdmap0[i]=j;
            flag=true;
        }
    }
    if(!flag)
    {
        cout << "There is no alignment between the two structures! "
             << "Program stop with no result!" << endl;
        TM1=TM2=TM3=TM4=TM5=0;
        return 1;
    }


    //********************************************************************//
    //    Detailed TMscore search engine --> prepare for final TMscore    //
    //********************************************************************//
    //run detailed TMscore search engine for the best alignment, and
    //extract the best rotation matrix (t, u) for the best alignment
    simplify_step=1;
    if (fast_opt) simplify_step=40;
    score_sum_method=8;
    TM = detailed_search_standard(r1, r2, xtm, ytm, xt, xa, ya, xlen, ylen,
        invmap0, t, u, simplify_step, score_sum_method, local_d0_search,
        false, Lnorm, score_d8, d0);
    
    double rmsd;
    simplify_step=1;
    score_sum_method=0;
    double Lnorm_0=ylen;

    //select pairs with dis<d8 for final TMscore computation and output alignment
    int k=0;
    int *m1, *m2;
    double d;
    m1=new int[xlen]; //alignd index in x
    m2=new int[ylen]; //alignd index in y
    copy_t_u(t, u, t0, u0);
    
    //****************************************//
    //              Final TMscore 1           //
    //****************************************//

    do_rotation(xa, xt, xlen, t, u);
    k=0;
    n_ali=0;
    for (i=0; i<xlen; i++)
    {
        j=fwdmap0[i];
        if(j>=0)//aligned
        {
            n_ali++;
            d=sqrt(dist(&xt[i][0], &ya[j][0]));
            if (d <= score_d8)
            {
                m1[k]=i;
                m2[k]=j;

                xtm[k][0]=xa[i][0];
                xtm[k][1]=xa[i][1];
                xtm[k][2]=xa[i][2];

                ytm[k][0]=ya[j][0];
                ytm[k][1]=ya[j][1];
                ytm[k][2]=ya[j][2];

                r1[k][0] = xt[i][0];
                r1[k][1] = xt[i][1];
                r1[k][2] = xt[i][2];
                r2[k][0] = ya[j][0];
                r2[k][1] = ya[j][1];
                r2[k][2] = ya[j][2];

                k++;
            }
            else fwdmap0[i]=-1;
        }
    }
    n_ali8=k;

    Kabsch(r1, r2, n_ali8, 0, &rmsd0, t, u);// rmsd0 is used for final output, only recalculate rmsd0, not t & u
    rmsd0 = sqrt(rmsd0 / n_ali8);
    
    //normalized by length of structure A
    parameter_set4final(xlen+0.0, D0_MIN, Lnorm, d0, d0_search, mol_type);
    d0B=d0;
    local_d0_search = d0_search;
    TM2 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t, u, simplify_step,
        score_sum_method, &rmsd, local_d0_search, Lnorm, score_d8, d0);

    //****************************************//
    //              Final TMscore 2           //
    //****************************************//
    
    do_rotation(xa, xt, xlen, t0, u0);
    k=0;
    for (j=0; j<ylen; j++)
    {
        i=invmap0[j];
        if(i>=0)//aligned
        {
            d=sqrt(dist(&xt[i][0], &ya[j][0]));
            if (d <= score_d8)
            {
                m1[k]=i;
                m2[k]=j;

                xtm[k][0]=xa[i][0];
                xtm[k][1]=xa[i][1];
                xtm[k][2]=xa[i][2];

                ytm[k][0]=ya[j][0];
                ytm[k][1]=ya[j][1];
                ytm[k][2]=ya[j][2];

                r1[k][0] = xt[i][0];
                r1[k][1] = xt[i][1];
                r1[k][2] = xt[i][2];
                r2[k][0] = ya[j][0];
                r2[k][1] = ya[j][1];
                r2[k][2] = ya[j][2];

                k++;
            }
            else invmap[j]=invmap0[j]=-1;
        }
    }

    //normalized by length of structure B
    parameter_set4final(Lnorm_0, D0_MIN, Lnorm, d0, d0_search, mol_type);
    d0A=d0;
    d0_0=d0A;
    local_d0_search = d0_search;
    TM1 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0, simplify_step,
        score_sum_method, &rmsd, local_d0_search, Lnorm, score_d8, d0);
    TM_0 = TM1;

    if (a_opt>0)
    {
        //normalized by average length of structures A, B
        Lnorm_0=(xlen+ylen)*0.5;
        parameter_set4final(Lnorm_0, D0_MIN, Lnorm, d0, d0_search, mol_type);
        d0a=d0;
        d0_0=d0a;
        local_d0_search = d0_search;

        TM3 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0,
            simplify_step, score_sum_method, &rmsd, local_d0_search, Lnorm,
            score_d8, d0);
        TM_0=TM3;
    }
    if (u_opt)
    {
        //normalized by user assigned length
        parameter_set4final(Lnorm_ass, D0_MIN, Lnorm,
            d0, d0_search, mol_type);
        d0u=d0;
        d0_0=d0u;
        Lnorm_0=Lnorm_ass;
        local_d0_search = d0_search;
        TM4 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0,
            simplify_step, score_sum_method, &rmsd, local_d0_search, Lnorm,
            score_d8, d0);
        TM_0=TM4;
    }
    if (d_opt)
    {
        //scaled by user assigned d0
        parameter_set4scale(ylen, d0_scale, Lnorm, d0, d0_search);
        d0_out=d0_scale;
        d0_0=d0_scale;
        //Lnorm_0=ylen;
        local_d0_search = d0_search;
        TM5 = TMscore8_search(r1, r2, xtm, ytm, xt, n_ali8, t0, u0,
            simplify_step, score_sum_method, &rmsd, local_d0_search, Lnorm,
            score_d8, d0);
        TM_0=TM5;
    }

    /* derive alignment from superposition */
    int ali_len=xlen+ylen;
    for (j=0;j<ylen;j++) ali_len-=(invmap0[j]>=0);
    seqxA.assign(ali_len,'-');
    seqM.assign( ali_len,' ');
    seqyA.assign(ali_len,'-');
    
    //do_rotation(xa, xt, xlen, t, u);
    do_rotation(xa, xt, xlen, t0, u0);

    Liden=0;
    //double SO=0;
    for (j=0;j<ylen;j++)
    {
        seqyA[j]=seqy[j];
        i=invmap0[j];
        dist_list[j]=-1;
        if (i<0) continue;
        d=sqrt(dist(xt[i], ya[j]));
        if (d<d0_out) seqM[j]=':';
        else seqM[j]='.';
        dist_list[j]=d;
        //SO+=(d<3.5);
        seqxA[j]=seqx[i];
        Liden+=(seqx[i]==seqy[j]);
    }
    //SO/=getmin(xlen,ylen);
    k=0;
    for (i=0;i<xlen;i++)
    {
        j=fwdmap0[i];
        if (j>=0) continue;
        seqxA[ylen+k]=seqx[i];
        k++;
    }
    //cout<<n_ali8<<'\t'
        //<<rmsd0<<'\t'
        //<<100.*SO<<endl;


    /* clean up */
    DeleteArray(&score, xlen+1);
    DeleteArray(&scoret,ylen+1);
    DeleteArray(&path,maxlen+1);
    DeleteArray(&val, maxlen+1);
    DeleteArray(&xtm, minlen);
    DeleteArray(&ytm, minlen);
    DeleteArray(&xt,xlen);
    DeleteArray(&yt,ylen);
    DeleteArray(&r1, minlen);
    DeleteArray(&r2, minlen);
    delete[]invmap0;
    delete[]fwdmap0;
    delete[]m1;
    delete[]m2;
    return 0;
}

/* USalign.cpp */
/* command line argument parsing and document of US-align main program */

void print_version()
{
    cout << 
"\n"
" ********************************************************************\n"
" * US-align (Version 20220924)                                      *\n"
" * Universal Structure Alignment of Proteins and Nucleic Acids      *\n"
" * Reference: C Zhang, M Shine, AM Pyle, Y Zhang. (2022) Nat Methods*\n"
" * Please email comments and suggestions to yangzhanglab@umich.edu  *\n"
" ********************************************************************"
    << endl;
}

void print_extra_help()
{
    cout <<
"Additional options:\n"
"      -v  Print the version of US-align\n"
"\n"
"      -a  TM-score normalized by the average length of two structures\n"
"          T or F, (default F). -a does not change the final alignment.\n"
"\n"
"   -fast  Fast but slightly inaccurate alignment\n"
"\n"
"    -dir  Perform all-against-all alignment among the list of PDB\n"
"          chains listed by 'chain_list' under 'chain_folder'. Note\n"
"          that the slash is necessary.\n"
"          $ USalign -dir chain_folder/ chain_list\n"
"\n"
"   -dir1  Use chain2 to search a list of PDB chains listed by 'chain1_list'\n"
"          under 'chain1_folder'. Note that the slash is necessary.\n"
"          $ USalign -dir1 chain1_folder/ chain1_list chain2\n"
"\n"
"   -dir2  Use chain1 to search a list of PDB chains listed by 'chain2_list'\n"
"          under 'chain2_folder'\n"
"          $ USalign chain1 -dir2 chain2_folder/ chain2_list\n"
"\n"
" -suffix  (Only when -dir1 and/or -dir2 are set, default is empty)\n"
"          add file name suffix to files listed by chain1_list or chain2_list\n"
"\n"
"   -atom  4-character atom name used to represent a residue.\n"
"          Default is \" C3'\" for RNA/DNA and \" CA \" for proteins\n"
"          (note the spaces before and after CA).\n"
"\n"
"  -split  Whether to split PDB file into multiple chains\n"
"           0: treat the whole structure as one single chain\n"
"           1: treat each MODEL as a separate chain\n"
"           2: (default) treat each chain as a separate chain\n"
"\n"
" -outfmt  Output format\n"
"           0: (default) full output\n"
"           1: fasta format compact output\n"
"           2: tabular format very compact output\n"
"          -1: full output, but without version or citation information\n"
"\n"
"  -TMcut  -1: (default) do not consider TMcut\n"
"          Values in [0.5,1): Do not proceed with TM-align for this\n"
"          structure pair if TM-score is unlikely to reach TMcut.\n"
"          TMcut is normalized as set by -a option:\n"
"          -2: normalized by longer structure length\n"
"          -1: normalized by shorter structure length\n"
"           0: (default, same as F) normalized by second structure\n"
"           1: same as T, normalized by average structure length\n"
"\n"
" -mirror  Whether to align the mirror image of input structure\n"
"           0: (default) do not align mirrored structure\n"
"           1: align mirror of Structure_1 to origin Structure_2,\n"
"              which usually requires the '-het 1' option:\n"
"              $ USalign 4glu.pdb 3p9w.pdb -mirror 1 -het 1\n"
"\n"
"    -het  Whether to align residues marked as 'HETATM' in addition to 'ATOM  '\n"
"           0: (default) only align 'ATOM  ' residues\n"
"           1: align both 'ATOM  ' and 'HETATM' residues\n"
"           2: align both 'ATOM  ' and MSE residues\n"
"\n"
"   -full  Whether to show full pairwise alignment of individual chains for\n"
"          -mm 2 or 4. T or F, (default F)\n"
//"\n"
"   -se    Do not perform superposition. Useful for extracting alignment from\n"
"          superposed structure pairs\n"
"\n"
" -infmt1  Input format for structure_11\n"
" -infmt2  Input format for structure_2\n"
"          -1: (default) automatically detect PDB or PDBx/mmCIF format\n"
"           0: PDB format\n"
"           1: SPICKER format\n"
//"           2: xyz format\n"
"           3: PDBx/mmCIF format\n"
"\n"
"Advanced usage 1 (generate an image for a pair of superposed structures):\n"
"    USalign 1cpc.pdb 1mba.pdb -o sup\n"
"    pymol -c -d @sup_all_atm.pml -g sup_all_atm.png\n"
"\n"
"Advanced usage 2 (a quick search of query.pdb against I-TASSER PDB library):\n"
"    wget https://zhanggroup.org/library/PDB.tar.bz2\n"
"    tar -xjvf PDB.tar.bz2\n"
"    USalign query.pdb -dir2 PDB/ PDB/list -suffix .pdb -outfmt 2 -fast\n"
    <<endl;
}

void print_help(bool h_opt=false)
{
    print_version();
    cout <<
"\n"
"Usage: USalign PDB1.pdb PDB2.pdb [Options]\n"
"\n"
"Options:\n"
"    -mol  Type of molecule(s) to align.\n"
"          auto: (default) align both protein and nucleic acids.\n"
"          prot: only align proteins in a structure.\n"
"          RNA : only align RNA and DNA in a structure.\n"
"\n"
"     -mm  Multimeric alignment option:\n"
"          0: (default) alignment of two monomeric structures\n"
"          1: alignment of two multi-chain oligomeric structures\n"
"          2: alignment of individual chains to an oligomeric structure\n"
"             $ USalign -dir1 monomers/ list oligomer.pdb -ter 0 -mm 2\n"
"          3: alignment of circularly permuted structure\n"
"          4: alignment of multiple monomeric chains into a consensus alignment\n"
"             $ USalign -dir chains/ list -suffix .pdb -mm 4\n"
"          5: fully non-sequential (fNS) alignment\n"
"          6: semi-non-sequential (sNS) alignment\n"
"          To use -mm 1 or -mm 2, '-ter' option must be 0 or 1.\n"
"\n"
"    -ter  Number of chains to align.\n"
"          3: only align the first chain, or the first segment of the\n"
"             first chain as marked by the 'TER' string in PDB file\n"
"          2: (default) only align the first chain\n"
"          1: align all chains of the first model (recommended for aligning\n"
"             asymmetric units)\n"
"          0: align all chains from all models (recommended for aligning\n"
"             biological assemblies, i.e. biounits)\n"
"\n"
" -TMscore Whether to perform TM-score superposition without structure-based\n"
"          alignment. The same as -byresi.\n"
"          0: (default) sequence independent structure alignment\n"
"          1: superpose two structures by assuming that a pair of residues\n"
"             with the same residue index are equivalent between the two\n"
"             structures\n"
"          2: superpose two complex structures, assuming that a pair of\n"
"             residues with the same residue index and the same chain ID\n"
"             are equivalent between the two structures\n"
//"          3: (similar to TMscore '-c' option; used with -ter 0 or 1)\n"
//"             align by residue index and order of chain\n"
//"          4: sequence dependent alignment: perform Needleman-Wunsch\n"
//"             global sequence alignment, followed by TM-score superposition\n"
"          5: sequence dependent alignment: perform glocal sequence\n"
"             alignment followed by TM-score superposition.\n"
"             -byresi 5 is the same as -seq\n"
"          6: superpose two complex structures by first deriving optimal\n"
"             chain mapping, followed by TM-score superposition for residues\n"
"             with the same residue ID\n"
"\n"
"      -I  Use the final alignment specified by FASTA file 'align.txt'\n"
"\n"
"      -i  Use alignment specified by 'align.txt' as an initial alignment\n"
"\n"
"      -m  Output rotation matrix for superposition\n"
"\n"
"      -d  TM-score scaled by an assigned d0, e.g., '-d 3.5' reports MaxSub\n"
"          score, where d0 is 3.5 Angstrom. -d does not change final alignment.\n"
"\n"
"      -u  TM-score normalized by an assigned length. It should be >= length\n"
"          of protein to avoid TM-score >1. -u does not change final alignment.\n"
"\n"
"      -o  Output superposed structure1 to sup.* for PyMOL viewing.\n"
"          $ USalign structure1.pdb structure2.pdb -o sup\n"
"          $ pymol -d @sup.pml                # C-alpha trace aligned region\n"
"          $ pymol -d @sup_all.pml            # C-alpha trace whole chain\n"
"          $ pymol -d @sup_atm.pml            # full-atom aligned region\n"
"          $ pymol -d @sup_all_atm.pml        # full-atom whole chain\n"
"          $ pymol -d @sup_all_atm_lig.pml    # full-atom with all molecules\n"
"\n"
" -rasmol  Output superposed structure1 to sup.* for RasMol viewing.\n"
"          $ USalign structure1.pdb structure2.pdb -rasmol sup\n"
"          $ rasmol -script sup               # C-alpha trace aligned region\n"
"          $ rasmol -script sup_all           # C-alpha trace whole chain\n"
"          $ rasmol -script sup_atm           # full-atom aligned region\n"
"          $ rasmol -script sup_all_atm       # full-atom whole chain\n"
"          $ rasmol -script sup_all_atm_lig   # full-atom with all molecules\n"
"\n"
//"      -h  Print the full help message, including additional options\n"
//"\n"
"Example usages ('gunzip' program is needed to read .gz compressed files):\n"
"    USalign 101m.cif.gz 1mba.pdb             # pairwise monomeric protein alignment\n"
"    USalign 1qf6.cif 5yyn.pdb.gz -mol RNA    # pairwise monomeric RNA alignment\n"
"    USalign model.pdb native.pdb -TMscore 1  # calculate TM-score between two conformations of a monomer\n"
"    USalign 4v4a.cif 4v49.cif -mm 1 -ter 1   # oligomeric alignment for asymmetic units\n"
"    USalign 3ksc.pdb1 4lej.pdb1 -mm 1 -ter 0 # oligomeric alignment for biological units\n"
"    USalign 1ajk.pdb.gz 2ayh.pdb.gz -mm 3    # circular permutation alignment\n"
    <<endl;

    //if (h_opt) 
        print_extra_help();

    exit(EXIT_SUCCESS);
}

/* TMalign, RNAalign, CPalign, TMscore */
int TMalign(string &xname, string &yname, const string &fname_super,
    const string &fname_lign, const string &fname_matrix,
    vector<string> &sequence, const double Lnorm_ass, const double d0_scale,
    const bool m_opt, const int  i_opt, const int o_opt, const int a_opt,
    const bool u_opt, const bool d_opt, const double TMcut,
    const int infmt1_opt, const int infmt2_opt, const int ter_opt,
    const int split_opt, const int outfmt_opt, const bool fast_opt,
    const int cp_opt, const int mirror_opt, const int het_opt,
    const string &atom_opt, const string &mol_opt, const string &dir_opt,
    const string &dir1_opt, const string &dir2_opt, const int byresi_opt,
    const vector<string> &chain1_list, const vector<string> &chain2_list,
    const bool se_opt)
{
    /* declare previously global variables */
    vector<vector<string> >PDB_lines1; // text of chain1
    vector<vector<string> >PDB_lines2; // text of chain2
    vector<int> mol_vec1;              // molecule type of chain1, RNA if >0
    vector<int> mol_vec2;              // molecule type of chain2, RNA if >0
    vector<string> chainID_list1;      // list of chainID1
    vector<string> chainID_list2;      // list of chainID2
    int    i,j;                // file index
    int    chain_i,chain_j;    // chain index
    int    r;                  // residue index
    int    xlen, ylen;         // chain length
    int    xchainnum,ychainnum;// number of chains in a PDB file
    char   *seqx, *seqy;       // for the protein sequence 
    char   *secx, *secy;       // for the secondary structure 
    double **xa, **ya;         // for input vectors xa[0...xlen-1][0..2] and
                               // ya[0...ylen-1][0..2], in general,
                               // ya is regarded as native structure 
                               // --> superpose xa onto ya
    vector<string> resi_vec1;  // residue index for chain1
    vector<string> resi_vec2;  // residue index for chain2
    int read_resi=byresi_opt;  // whether to read residue index
    if (byresi_opt==0 && o_opt) read_resi=2;

    /* loop over file names */
    for (i=0;i<chain1_list.size();i++)
    {
        /* parse chain 1 */
        xname=chain1_list[i];
        xchainnum=get_PDB_lines(xname, PDB_lines1, chainID_list1,
            mol_vec1, ter_opt, infmt1_opt, atom_opt, split_opt, het_opt);
        if (!xchainnum)
        {
            cerr<<"Warning! Cannot parse file: "<<xname
                <<". Chain number 0."<<endl;
            continue;
        }
        for (chain_i=0;chain_i<xchainnum;chain_i++)
        {
            xlen=PDB_lines1[chain_i].size();
            if (mol_opt=="RNA") mol_vec1[chain_i]=1;
            else if (mol_opt=="protein") mol_vec1[chain_i]=-1;
            if (!xlen)
            {
                cerr<<"Warning! Cannot parse file: "<<xname
                    <<". Chain length 0."<<endl;
                continue;
            }
            else if (xlen<3)
            {
                cerr<<"Sequence is too short <3!: "<<xname<<endl;
                continue;
            }
            NewArray(&xa, xlen, 3);
            seqx = new char[xlen + 1];
            secx = new char[xlen + 1];
            xlen = read_PDB(PDB_lines1[chain_i], xa, seqx, 
                resi_vec1, read_resi);
            if (mirror_opt) for (r=0;r<xlen;r++) xa[r][2]=-xa[r][2];
            if (mol_vec1[chain_i]>0) make_sec(seqx,xa, xlen, secx,atom_opt);
            else make_sec(xa, xlen, secx); // secondary structure assignment

            for (j=(dir_opt.size()>0)*(i+1);j<chain2_list.size();j++)
            {
                /* parse chain 2 */
                if (PDB_lines2.size()==0)
                {
                    yname=chain2_list[j];
                    ychainnum=get_PDB_lines(yname, PDB_lines2, chainID_list2,
                        mol_vec2, ter_opt, infmt2_opt, atom_opt, split_opt,
                        het_opt);
                    if (!ychainnum)
                    {
                        cerr<<"Warning! Cannot parse file: "<<yname
                            <<". Chain number 0."<<endl;
                        continue;
                    }
                }
                for (chain_j=0;chain_j<ychainnum;chain_j++)
                {
                    ylen=PDB_lines2[chain_j].size();
                    if (mol_opt=="RNA") mol_vec2[chain_j]=1;
                    else if (mol_opt=="protein") mol_vec2[chain_j]=-1;
                    if (!ylen)
                    {
                        cerr<<"Warning! Cannot parse file: "<<yname
                            <<". Chain length 0."<<endl;
                        continue;
                    }
                    else if (ylen<3)
                    {
                        cerr<<"Sequence is too short <3!: "<<yname<<endl;
                        continue;
                    }
                    NewArray(&ya, ylen, 3);
                    seqy = new char[ylen + 1];
                    secy = new char[ylen + 1];
                    ylen = read_PDB(PDB_lines2[chain_j], ya, seqy,
                        resi_vec2, read_resi);
                    if (mol_vec2[chain_j]>0)
                         make_sec(seqy, ya, ylen, secy, atom_opt);
                    else make_sec(ya, ylen, secy);

                    if (byresi_opt) extract_aln_from_resi(sequence,
                        seqx,seqy,resi_vec1,resi_vec2,byresi_opt);

                    /* declare variable specific to this pair of TMalign */
                    double t0[3], u0[3][3];
                    double TM1, TM2;
                    double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
                    double d0_0, TM_0;
                    double d0A, d0B, d0u, d0a;
                    double d0_out=5.0;
                    string seqM, seqxA, seqyA;// for output alignment
                    double rmsd0 = 0.0;
                    int L_ali;                // Aligned length in standard_TMscore
                    double Liden=0;
                    double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
                    int n_ali=0;
                    int n_ali8=0;
                    bool force_fast_opt=(getmin(xlen,ylen)>1500)?true:fast_opt;

                    /* entry function for structure alignment */
                    if (cp_opt) CPalign_main(
                        xa, ya, seqx, seqy, secx, secy,
                        t0, u0, TM1, TM2, TM3, TM4, TM5,
                        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                        seqM, seqxA, seqyA,
                        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                        xlen, ylen, sequence, Lnorm_ass, d0_scale,
                        i_opt, a_opt, u_opt, d_opt, force_fast_opt,
                        mol_vec1[chain_i]+mol_vec2[chain_j],TMcut);
                    else if (se_opt)
                    {
                        int *invmap = new int[ylen+1];
                        u0[0][0]=u0[1][1]=u0[2][2]=1;
                        u0[0][1]=         u0[0][2]=
                        u0[1][0]=         u0[1][2]=
                        u0[2][0]=         u0[2][1]=
                        t0[0]   =t0[1]   =t0[2]   =0;
                        se_main(
                            xa, ya, seqx, seqy, TM1, TM2, TM3, TM4, TM5,
                            d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                            seqM, seqxA, seqyA,
                            rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                            xlen, ylen, sequence, Lnorm_ass, d0_scale,
                            i_opt, a_opt, u_opt, d_opt,
                            mol_vec1[chain_i]+mol_vec2[chain_j], 
                            outfmt_opt, invmap);
                        if (outfmt_opt>=2) 
                        {
                            Liden=L_ali=0;
                            int r1,r2;
                            for (r2=0;r2<ylen;r2++)
                            {
                                r1=invmap[r2];
                                if (r1<0) continue;
                                L_ali+=1;
                                Liden+=(seqx[r1]==seqy[r2]);
                            }
                        }
                        delete [] invmap;
                    }
                    else TMalign_main(
                        xa, ya, seqx, seqy, secx, secy,
                        t0, u0, TM1, TM2, TM3, TM4, TM5,
                        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                        seqM, seqxA, seqyA,
                        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                        xlen, ylen, sequence, Lnorm_ass, d0_scale,
                        i_opt, a_opt, u_opt, d_opt, force_fast_opt,
                        mol_vec1[chain_i]+mol_vec2[chain_j],TMcut);

                    /* print result */
                    if (outfmt_opt==0) print_version();
                    if (cp_opt) output_cp(
                        xname.substr(dir1_opt.size()+dir_opt.size()),
                        yname.substr(dir2_opt.size()+dir_opt.size()),
                        seqxA,seqyA,outfmt_opt);
                    output_results(
                        xname.substr(dir1_opt.size()+dir_opt.size()),
                        yname.substr(dir2_opt.size()+dir_opt.size()),
                        chainID_list1[chain_i], chainID_list2[chain_j],
                        xlen, ylen, t0, u0, TM1, TM2, TM3, TM4, TM5,
                        rmsd0, d0_out, seqM.c_str(),
                        seqxA.c_str(), seqyA.c_str(), Liden,
                        n_ali8, L_ali, TM_ali, rmsd_ali, TM_0, d0_0,
                        d0A, d0B, Lnorm_ass, d0_scale, d0a, d0u, 
                        (m_opt?fname_matrix:"").c_str(),
                        outfmt_opt, ter_opt, false, split_opt, o_opt,
                        fname_super, i_opt, a_opt, u_opt, d_opt, mirror_opt,
                        resi_vec1, resi_vec2);

                    /* Done! Free memory */
                    seqM.clear();
                    seqxA.clear();
                    seqyA.clear();
                    DeleteArray(&ya, ylen);
                    delete [] seqy;
                    delete [] secy;
                    resi_vec2.clear();
                } // chain_j
                if (chain2_list.size()>1)
                {
                    yname.clear();
                    for (chain_j=0;chain_j<ychainnum;chain_j++)
                        PDB_lines2[chain_j].clear();
                    PDB_lines2.clear();
                    chainID_list2.clear();
                    mol_vec2.clear();
                }
            } // j
            PDB_lines1[chain_i].clear();
            DeleteArray(&xa, xlen);
            delete [] seqx;
            delete [] secx;
            resi_vec1.clear();
        } // chain_i
        xname.clear();
        PDB_lines1.clear();
        chainID_list1.clear();
        mol_vec1.clear();
    } // i
    if (chain2_list.size()==1)
    {
        yname.clear();
        for (chain_j=0;chain_j<ychainnum;chain_j++)
            PDB_lines2[chain_j].clear();
        PDB_lines2.clear();
        resi_vec2.clear();
        chainID_list2.clear();
        mol_vec2.clear();
    }
    return 0;
}

/* MMalign if more than two chains. TMalign if only one chain */
int MMalign(const string &xname, const string &yname,
    const string &fname_super, const string &fname_lign,
    const string &fname_matrix, vector<string> &sequence,
    const double d0_scale, const bool m_opt, const int o_opt,
    const int a_opt, const bool d_opt, const bool full_opt,
    const double TMcut, const int infmt1_opt, const int infmt2_opt,
    const int ter_opt, const int split_opt, const int outfmt_opt,
    bool fast_opt, const int mirror_opt, const int het_opt,
    const string &atom_opt, const string &mol_opt,
    const string &dir1_opt, const string &dir2_opt,
    const vector<string> &chain1_list, const vector<string> &chain2_list,
    const int byresi_opt)
{
    /* declare previously global variables */
    vector<vector<vector<double> > > xa_vec; // structure of complex1
    vector<vector<vector<double> > > ya_vec; // structure of complex2
    vector<vector<char> >seqx_vec; // sequence of complex1
    vector<vector<char> >seqy_vec; // sequence of complex2
    vector<vector<char> >secx_vec; // secondary structure of complex1
    vector<vector<char> >secy_vec; // secondary structure of complex2
    vector<int> mol_vec1;          // molecule type of complex1, RNA if >0
    vector<int> mol_vec2;          // molecule type of complex2, RNA if >0
    vector<string> chainID_list1;  // list of chainID1
    vector<string> chainID_list2;  // list of chainID2
    vector<int> xlen_vec;          // length of complex1
    vector<int> ylen_vec;          // length of complex2
    int    i,j;                    // chain index
    int    xlen, ylen;             // chain length
    double **xa, **ya;             // structure of single chain
    char   *seqx, *seqy;           // for the protein sequence 
    char   *secx, *secy;           // for the secondary structure 
    int    xlen_aa,ylen_aa;        // total length of protein
    int    xlen_na,ylen_na;        // total length of RNA/DNA
    vector<string> resi_vec1;  // residue index for chain1
    vector<string> resi_vec2;  // residue index for chain2

    /* parse complex */
    parse_chain_list(chain1_list, xa_vec, seqx_vec, secx_vec, mol_vec1,
        xlen_vec, chainID_list1, ter_opt, split_opt, mol_opt, infmt1_opt,
        atom_opt, mirror_opt, het_opt, xlen_aa, xlen_na, o_opt, resi_vec1);
    if (xa_vec.size()==0) PrintErrorAndQuit("ERROR! 0 chain in complex 1");
    parse_chain_list(chain2_list, ya_vec, seqy_vec, secy_vec, mol_vec2,
        ylen_vec, chainID_list2, ter_opt, split_opt, mol_opt, infmt2_opt,
        atom_opt, 0, het_opt, ylen_aa, ylen_na, o_opt, resi_vec2);
    if (ya_vec.size()==0) PrintErrorAndQuit("ERROR! 0 chain in complex 2");
    int len_aa=getmin(xlen_aa,ylen_aa);
    int len_na=getmin(xlen_na,ylen_na);
    if (a_opt)
    {
        len_aa=(xlen_aa+ylen_aa)/2;
        len_na=(xlen_na+ylen_na)/2;
    }
    int i_opt=0;
    if (byresi_opt) i_opt=3;

    /* perform monomer alignment if there is only one chain */
    if (xa_vec.size()==1 && ya_vec.size()==1)
    {
        xlen = xlen_vec[0];
        ylen = ylen_vec[0];
        seqx = new char[xlen+1];
        seqy = new char[ylen+1];
        secx = new char[xlen+1];
        secy = new char[ylen+1];
        NewArray(&xa, xlen, 3);
        NewArray(&ya, ylen, 3);
        copy_chain_data(xa_vec[0],seqx_vec[0],secx_vec[0], xlen,xa,seqx,secx);
        copy_chain_data(ya_vec[0],seqy_vec[0],secy_vec[0], ylen,ya,seqy,secy);
        
        /* declare variable specific to this pair of TMalign */
        double t0[3], u0[3][3];
        double TM1, TM2;
        double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
        double d0_0, TM_0;
        double d0A, d0B, d0u, d0a;
        double d0_out=5.0;
        string seqM, seqxA, seqyA;// for output alignment
        double rmsd0 = 0.0;
        int L_ali;                // Aligned length in standard_TMscore
        double Liden=0;
        double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
        int n_ali=0;
        int n_ali8=0;
        
        if (byresi_opt) extract_aln_from_resi(sequence,
            seqx,seqy,resi_vec1,resi_vec2,byresi_opt);

        /* entry function for structure alignment */
        TMalign_main(xa, ya, seqx, seqy, secx, secy,
            t0, u0, TM1, TM2, TM3, TM4, TM5,
            d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
            seqM, seqxA, seqyA,
            rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
            xlen, ylen, sequence, 0, d0_scale,
            i_opt, a_opt, false, d_opt, fast_opt,
            mol_vec1[0]+mol_vec2[0],TMcut);

        /* print result */
        output_results(
            xname.substr(dir1_opt.size()),
            yname.substr(dir2_opt.size()),
            chainID_list1[0], chainID_list2[0],
            xlen, ylen, t0, u0, TM1, TM2, TM3, TM4, TM5, rmsd0, d0_out,
            seqM.c_str(), seqxA.c_str(), seqyA.c_str(), Liden,
            n_ali8, L_ali, TM_ali, rmsd_ali, TM_0, d0_0, d0A, d0B,
            0, d0_scale, d0a, d0u, (m_opt?fname_matrix:"").c_str(),
            outfmt_opt, ter_opt, true, split_opt, o_opt, fname_super,
            0, a_opt, false, d_opt, mirror_opt, resi_vec1, resi_vec2);

        /* clean up */
        seqM.clear();
        seqxA.clear();
        seqyA.clear();
        delete[]seqx;
        delete[]seqy;
        delete[]secx;
        delete[]secy;
        DeleteArray(&xa,xlen);
        DeleteArray(&ya,ylen);

        vector<vector<vector<double> > >().swap(xa_vec); // structure of complex1
        vector<vector<vector<double> > >().swap(ya_vec); // structure of complex2
        vector<vector<char> >().swap(seqx_vec); // sequence of complex1
        vector<vector<char> >().swap(seqy_vec); // sequence of complex2
        vector<vector<char> >().swap(secx_vec); // secondary structure of complex1
        vector<vector<char> >().swap(secy_vec); // secondary structure of complex2
        mol_vec1.clear();       // molecule type of complex1, RNA if >0
        mol_vec2.clear();       // molecule type of complex2, RNA if >0
        chainID_list1.clear();  // list of chainID1
        chainID_list2.clear();  // list of chainID2
        xlen_vec.clear();       // length of complex1
        ylen_vec.clear();       // length of complex2
        return 0;
    }

    /* declare TM-score tables */
    int chain1_num=xa_vec.size();
    int chain2_num=ya_vec.size();
    vector<string> tmp_str_vec(chain2_num,"");
    double **TMave_mat;
    double **ut_mat; // rotation matrices for all-against-all alignment
    int ui,uj,ut_idx;
    NewArray(&TMave_mat,chain1_num,chain2_num);
    NewArray(&ut_mat,chain1_num*chain2_num,4*3);
    vector<vector<string> >seqxA_mat(chain1_num,tmp_str_vec);
    vector<vector<string> > seqM_mat(chain1_num,tmp_str_vec);
    vector<vector<string> >seqyA_mat(chain1_num,tmp_str_vec);

    double maxTMmono=-1;
    int maxTMmono_i,maxTMmono_j;

    /* get all-against-all alignment */
    if (len_aa+len_na>500) fast_opt=true;
    for (i=0;i<chain1_num;i++)
    {
        xlen=xlen_vec[i];
        if (xlen<3)
        {
            for (j=0;j<chain2_num;j++) TMave_mat[i][j]=-1;
            continue;
        }
        seqx = new char[xlen+1];
        secx = new char[xlen+1];
        NewArray(&xa, xlen, 3);
        copy_chain_data(xa_vec[i],seqx_vec[i],secx_vec[i],
            xlen,xa,seqx,secx);

        for (j=0;j<chain2_num;j++)
        {
            ut_idx=i*chain2_num+j;
            for (ui=0;ui<4;ui++)
                for (uj=0;uj<3;uj++) ut_mat[ut_idx][ui*3+uj]=0;
            ut_mat[ut_idx][0]=1;
            ut_mat[ut_idx][4]=1;
            ut_mat[ut_idx][8]=1;

            if (mol_vec1[i]*mol_vec2[j]<0) //no protein-RNA alignment
            {
                TMave_mat[i][j]=-1;
                continue;
            }

            ylen=ylen_vec[j];
            if (ylen<3)
            {
                TMave_mat[i][j]=-1;
                continue;
            }
            seqy = new char[ylen+1];
            secy = new char[ylen+1];
            NewArray(&ya, ylen, 3);
            copy_chain_data(ya_vec[j],seqy_vec[j],secy_vec[j],
                ylen,ya,seqy,secy);

            /* declare variable specific to this pair of TMalign */
            double t0[3], u0[3][3];
            double TM1, TM2;
            double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
            double d0_0, TM_0;
            double d0A, d0B, d0u, d0a;
            double d0_out=5.0;
            string seqM, seqxA, seqyA;// for output alignment
            double rmsd0 = 0.0;
            int L_ali;                // Aligned length in standard_TMscore
            double Liden=0;
            double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
            int n_ali=0;
            int n_ali8=0;

            int Lnorm_tmp=len_aa;
            if (mol_vec1[i]+mol_vec2[j]>0) Lnorm_tmp=len_na;
            
            if (byresi_opt)
            {
                int total_aln=extract_aln_from_resi(sequence,
                    seqx,seqy,resi_vec1,resi_vec2,xlen_vec,ylen_vec, i, j);
                seqxA_mat[i][j]=sequence[0];
                seqyA_mat[i][j]=sequence[1];
                if (total_aln>xlen+ylen-3)
                {
                    for (ui=0;ui<3;ui++) for (uj=0;uj<3;uj++) 
                        ut_mat[ut_idx][ui*3+uj]=(ui==uj)?1:0;
                    for (uj=0;uj<3;uj++) ut_mat[ut_idx][9+uj]=0;
                    TMave_mat[i][j]=0;
                    seqM.clear();
                    seqxA.clear();
                    seqyA.clear();

                    delete[]seqy;
                    delete[]secy;
                    DeleteArray(&ya,ylen);
                    continue;
                }
            }

            /* entry function for structure alignment */
            TMalign_main(xa, ya, seqx, seqy, secx, secy,
                t0, u0, TM1, TM2, TM3, TM4, TM5,
                d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                seqM, seqxA, seqyA,
                rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                xlen, ylen, sequence, Lnorm_tmp, d0_scale,
                i_opt, false, true, false, fast_opt,
                mol_vec1[i]+mol_vec2[j],TMcut);

            /* store result */
            for (ui=0;ui<3;ui++)
                for (uj=0;uj<3;uj++) ut_mat[ut_idx][ui*3+uj]=u0[ui][uj];
            for (uj=0;uj<3;uj++) ut_mat[ut_idx][9+uj]=t0[uj];
            seqxA_mat[i][j]=seqxA;
            seqyA_mat[i][j]=seqyA;
            TMave_mat[i][j]=TM4*Lnorm_tmp;
            if (TMave_mat[i][j]>maxTMmono)
            {
                maxTMmono=TMave_mat[i][j];
                maxTMmono_i=i;
                maxTMmono_j=j;
            }

            /* clean up */
            seqM.clear();
            seqxA.clear();
            seqyA.clear();

            delete[]seqy;
            delete[]secy;
            DeleteArray(&ya,ylen);
        }

        delete[]seqx;
        delete[]secx;
        DeleteArray(&xa,xlen);
    }

    /* calculate initial chain-chain assignment */
    int *assign1_list; // value is index of assigned chain2
    int *assign2_list; // value is index of assigned chain1
    assign1_list=new int[chain1_num];
    assign2_list=new int[chain2_num];
    double total_score=enhanced_greedy_search(TMave_mat, assign1_list,
        assign2_list, chain1_num, chain2_num);
    if (total_score<=0) PrintErrorAndQuit("ERROR! No assignable chain");

    /* refine alignment for large oligomers */
    int aln_chain_num=count_assign_pair(assign1_list,chain1_num);
    bool is_oligomer=(aln_chain_num>=3);
    if (aln_chain_num==2) // dimer alignment
    {
        int na_chain_num1,na_chain_num2,aa_chain_num1,aa_chain_num2;
        count_na_aa_chain_num(na_chain_num1,aa_chain_num1,mol_vec1);
        count_na_aa_chain_num(na_chain_num2,aa_chain_num2,mol_vec2);

        /* align protein-RNA hybrid dimer to another hybrid dimer */
        if (na_chain_num1==1 && na_chain_num2==1 && 
            aa_chain_num1==1 && aa_chain_num2==1) is_oligomer=false;
        /* align pure protein dimer or pure RNA dimer */
        else if ((getmin(na_chain_num1,na_chain_num2)==0 && 
                    aa_chain_num1==2 && aa_chain_num2==2) ||
                 (getmin(aa_chain_num1,aa_chain_num2)==0 && 
                    na_chain_num1==2 && na_chain_num2==2))
        {
            adjust_dimer_assignment(xa_vec,ya_vec,xlen_vec,ylen_vec,mol_vec1,
                mol_vec2,assign1_list,assign2_list,seqxA_mat,seqyA_mat);
            is_oligomer=false; // cannot refiner further
        }
        else is_oligomer=true; /* align oligomers to dimer */
    }

    if (aln_chain_num>=3 || is_oligomer) // oligomer alignment
    {
        /* extract centroid coordinates */
        double **xcentroids;
        double **ycentroids;
        NewArray(&xcentroids, chain1_num, 3);
        NewArray(&ycentroids, chain2_num, 3);
        double d0MM=getmin(
            calculate_centroids(xa_vec, chain1_num, xcentroids),
            calculate_centroids(ya_vec, chain2_num, ycentroids));

        /* refine enhanced greedy search with centroid superposition */
        //double het_deg=check_heterooligomer(TMave_mat, chain1_num, chain2_num);
        homo_refined_greedy_search(TMave_mat, assign1_list,
            assign2_list, chain1_num, chain2_num, xcentroids,
            ycentroids, d0MM, len_aa+len_na, ut_mat);
        hetero_refined_greedy_search(TMave_mat, assign1_list,
            assign2_list, chain1_num, chain2_num, xcentroids,
            ycentroids, d0MM, len_aa+len_na);
        
        /* clean up */
        DeleteArray(&xcentroids, chain1_num);
        DeleteArray(&ycentroids, chain2_num);
    }

    /* store initial assignment */
    int init_pair_num=count_assign_pair(assign1_list,chain1_num);
    int *assign1_init, *assign2_init;
    assign1_init=new int[chain1_num];
    assign2_init=new int[chain2_num];
    double **TMave_init;
    NewArray(&TMave_init,chain1_num,chain2_num);
    vector<vector<string> >seqxA_init(chain1_num,tmp_str_vec);
    vector<vector<string> >seqyA_init(chain1_num,tmp_str_vec);
    vector<string> sequence_init;
    copy_chain_assign_data(chain1_num, chain2_num, sequence_init,
        seqxA_mat,  seqyA_mat,  assign1_list, assign2_list, TMave_mat,
        seqxA_init, seqyA_init, assign1_init, assign2_init, TMave_init);

    /* perform iterative alignment */
    double max_total_score=0; // ignore old total_score because previous
                              // score was from monomeric chain superpositions
    int max_iter=5-(int)((len_aa+len_na)/200);
    if (max_iter<2) max_iter=2;
    if (byresi_opt==0) MMalign_iter(max_total_score, max_iter, xa_vec, ya_vec,
        seqx_vec, seqy_vec, secx_vec, secy_vec, mol_vec1, mol_vec2, xlen_vec,
        ylen_vec, xa, ya, seqx, seqy, secx, secy, len_aa, len_na, chain1_num,
        chain2_num, TMave_mat, seqxA_mat, seqyA_mat, assign1_list, assign2_list,
        sequence, d0_scale, fast_opt);

    /* sometime MMalign_iter is even worse than monomer alignment */
    if (byresi_opt==0 && max_total_score<maxTMmono)
    {
        copy_chain_assign_data(chain1_num, chain2_num, sequence,
            seqxA_init, seqyA_init, assign1_init, assign2_init, TMave_init,
            seqxA_mat, seqyA_mat, assign1_list, assign2_list, TMave_mat);
        for (i=0;i<chain1_num;i++)
        {
            if (i!=maxTMmono_i) assign1_list[i]=-1;
            else assign1_list[i]=maxTMmono_j;
        }
        for (j=0;j<chain2_num;j++)
        {
            if (j!=maxTMmono_j) assign2_list[j]=-1;
            else assign2_list[j]=maxTMmono_i;
        }
        sequence[0]=seqxA_mat[maxTMmono_i][maxTMmono_j];
        sequence[1]=seqyA_mat[maxTMmono_i][maxTMmono_j];
        max_total_score=maxTMmono;
        MMalign_iter(max_total_score, max_iter, xa_vec, ya_vec, seqx_vec, seqy_vec,
            secx_vec, secy_vec, mol_vec1, mol_vec2, xlen_vec, ylen_vec,
            xa, ya, seqx, seqy, secx, secy, len_aa, len_na, chain1_num, chain2_num,
            TMave_mat, seqxA_mat, seqyA_mat, assign1_list, assign2_list, sequence,
            d0_scale, fast_opt);
    }

    /* perform cross chain alignment
     * in some cases, this leads to dramatic improvement, esp for homodimer */
    int iter_pair_num=count_assign_pair(assign1_list,chain1_num);
    if (iter_pair_num>=init_pair_num) copy_chain_assign_data(
        chain1_num, chain2_num, sequence_init,
        seqxA_mat, seqyA_mat, assign1_list, assign2_list, TMave_mat,
        seqxA_init, seqyA_init, assign1_init,  assign2_init,  TMave_init);
    double max_total_score_cross=max_total_score;
    if (byresi_opt==0 && len_aa+len_na<10000)
    {
        MMalign_dimer(max_total_score_cross, xa_vec, ya_vec, seqx_vec, seqy_vec,
            secx_vec, secy_vec, mol_vec1, mol_vec2, xlen_vec, ylen_vec,
            xa, ya, seqx, seqy, secx, secy, len_aa, len_na, chain1_num, chain2_num,
            TMave_init, seqxA_init, seqyA_init, assign1_init, assign2_init,
            sequence_init, d0_scale, fast_opt);
        if (max_total_score_cross>max_total_score) 
        {
            max_total_score=max_total_score_cross;
            copy_chain_assign_data(chain1_num, chain2_num, sequence,
                seqxA_init, seqyA_init, assign1_init, assign2_init, TMave_init,
                seqxA_mat,  seqyA_mat,  assign1_list, assign2_list, TMave_mat);
        }
    } 

    /* final alignment */
    if (outfmt_opt==0) print_version();
    MMalign_final(xname.substr(dir1_opt.size()), yname.substr(dir2_opt.size()),
        chainID_list1, chainID_list2,
        fname_super, fname_lign, fname_matrix,
        xa_vec, ya_vec, seqx_vec, seqy_vec,
        secx_vec, secy_vec, mol_vec1, mol_vec2, xlen_vec, ylen_vec,
        xa, ya, seqx, seqy, secx, secy, len_aa, len_na,
        chain1_num, chain2_num, TMave_mat,
        seqxA_mat, seqM_mat, seqyA_mat, assign1_list, assign2_list, sequence,
        d0_scale, m_opt, o_opt, outfmt_opt, ter_opt, split_opt,
        a_opt, d_opt, fast_opt, full_opt, mirror_opt, resi_vec1, resi_vec2);

    /* clean up everything */
    delete [] assign1_list;
    delete [] assign2_list;
    DeleteArray(&TMave_mat,chain1_num);
    DeleteArray(&ut_mat,   chain1_num*chain2_num);
    vector<vector<string> >().swap(seqxA_mat);
    vector<vector<string> >().swap(seqM_mat);
    vector<vector<string> >().swap(seqyA_mat);
    vector<string>().swap(tmp_str_vec);

    delete [] assign1_init;
    delete [] assign2_init;
    DeleteArray(&TMave_init,chain1_num);
    vector<vector<string> >().swap(seqxA_init);
    vector<vector<string> >().swap(seqyA_init);

    vector<vector<vector<double> > >().swap(xa_vec); // structure of complex1
    vector<vector<vector<double> > >().swap(ya_vec); // structure of complex2
    vector<vector<char> >().swap(seqx_vec); // sequence of complex1
    vector<vector<char> >().swap(seqy_vec); // sequence of complex2
    vector<vector<char> >().swap(secx_vec); // secondary structure of complex1
    vector<vector<char> >().swap(secy_vec); // secondary structure of complex2
    mol_vec1.clear();       // molecule type of complex1, RNA if >0
    mol_vec2.clear();       // molecule type of complex2, RNA if >0
    vector<string>().swap(chainID_list1);  // list of chainID1
    vector<string>().swap(chainID_list2);  // list of chainID2
    xlen_vec.clear();       // length of complex1
    ylen_vec.clear();       // length of complex2
    vector<string> ().swap(resi_vec1);  // residue index for chain1
    vector<string> ().swap(resi_vec2);  // residue index for chain2
    return 1;
}


/* alignment individual chains to a complex. */
int MMdock(const string &xname, const string &yname, const string &fname_super, 
    const string &fname_matrix, vector<string> &sequence, const double Lnorm_ass,
    const double d0_scale, const bool m_opt, const int o_opt,
    const int a_opt, const bool u_opt, const bool d_opt,
    const double TMcut, const int infmt1_opt, const int infmt2_opt,
    const int ter_opt, const int split_opt, const int outfmt_opt,
    bool fast_opt, const int mirror_opt, const int het_opt,
    const string &atom_opt, const string &mol_opt,
    const string &dir1_opt, const string &dir2_opt,
    const vector<string> &chain1_list, const vector<string> &chain2_list)
{
    /* declare previously global variables */
    vector<vector<vector<double> > > xa_vec; // structure of complex1
    vector<vector<vector<double> > > ya_vec; // structure of complex2
    vector<vector<char> >seqx_vec; // sequence of complex1
    vector<vector<char> >seqy_vec; // sequence of complex2
    vector<vector<char> >secx_vec; // secondary structure of complex1
    vector<vector<char> >secy_vec; // secondary structure of complex2
    vector<int> mol_vec1;          // molecule type of complex1, RNA if >0
    vector<int> mol_vec2;          // molecule type of complex2, RNA if >0
    vector<string> chainID_list1;  // list of chainID1
    vector<string> chainID_list2;  // list of chainID2
    vector<int> xlen_vec;          // length of complex1
    vector<int> ylen_vec;          // length of complex2
    int    i,j;                    // chain index
    int    xlen, ylen;             // chain length
    double **xa, **ya;             // structure of single chain
    char   *seqx, *seqy;           // for the protein sequence 
    char   *secx, *secy;           // for the secondary structure 
    int    xlen_aa,ylen_aa;        // total length of protein
    int    xlen_na,ylen_na;        // total length of RNA/DNA
    vector<string> resi_vec1;  // residue index for chain1
    vector<string> resi_vec2;  // residue index for chain2

    /* parse complex */
    parse_chain_list(chain1_list, xa_vec, seqx_vec, secx_vec, mol_vec1,
        xlen_vec, chainID_list1, ter_opt, split_opt, mol_opt, infmt1_opt,
        atom_opt, mirror_opt, het_opt, xlen_aa, xlen_na, o_opt, resi_vec1);
    if (xa_vec.size()==0) PrintErrorAndQuit("ERROR! 0 individual chain");
    parse_chain_list(chain2_list, ya_vec, seqy_vec, secy_vec, mol_vec2,
        ylen_vec, chainID_list2, ter_opt, split_opt, mol_opt, infmt2_opt,
        atom_opt, 0, het_opt, ylen_aa, ylen_na, o_opt, resi_vec2);
    if (xa_vec.size()>ya_vec.size()) PrintErrorAndQuit(
        "ERROR! more individual chains to align than number of chains in complex template");
    int len_aa=getmin(xlen_aa,ylen_aa);
    int len_na=getmin(xlen_na,ylen_na);
    if (a_opt)
    {
        len_aa=(xlen_aa+ylen_aa)/2;
        len_na=(xlen_na+ylen_na)/2;
    }

    /* perform monomer alignment if there is only one chain */
    if (xa_vec.size()==1 && ya_vec.size()==1)
    {
        xlen = xlen_vec[0];
        ylen = ylen_vec[0];
        seqx = new char[xlen+1];
        seqy = new char[ylen+1];
        secx = new char[xlen+1];
        secy = new char[ylen+1];
        NewArray(&xa, xlen, 3);
        NewArray(&ya, ylen, 3);
        copy_chain_data(xa_vec[0],seqx_vec[0],secx_vec[0], xlen,xa,seqx,secx);
        copy_chain_data(ya_vec[0],seqy_vec[0],secy_vec[0], ylen,ya,seqy,secy);
        
        /* declare variable specific to this pair of TMalign */
        double t0[3], u0[3][3];
        double TM1, TM2;
        double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
        double d0_0, TM_0;
        double d0A, d0B, d0u, d0a;
        double d0_out=5.0;
        string seqM, seqxA, seqyA;// for output alignment
        double rmsd0 = 0.0;
        int L_ali;                // Aligned length in standard_TMscore
        double Liden=0;
        double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
        int n_ali=0;
        int n_ali8=0;

        /* entry function for structure alignment */
        TMalign_main(xa, ya, seqx, seqy, secx, secy,
            t0, u0, TM1, TM2, TM3, TM4, TM5,
            d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
            seqM, seqxA, seqyA,
            rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
            xlen, ylen, sequence, Lnorm_ass, d0_scale,
            0, a_opt, u_opt, d_opt, fast_opt,
            mol_vec1[0]+mol_vec2[0],TMcut);

        /* print result */
        output_results(
            xname.substr(dir1_opt.size()),
            yname.substr(dir2_opt.size()),
            chainID_list1[0], chainID_list2[0],
            xlen, ylen, t0, u0, TM1, TM2, TM3, TM4, TM5, rmsd0, d0_out,
            seqM.c_str(), seqxA.c_str(), seqyA.c_str(), Liden,
            n_ali8, L_ali, TM_ali, rmsd_ali, TM_0, d0_0, d0A, d0B,
            Lnorm_ass, d0_scale, d0a, d0u, (m_opt?fname_matrix:"").c_str(),
            (outfmt_opt==2?outfmt_opt:3), ter_opt, true, split_opt, o_opt, fname_super,
            0, a_opt, false, d_opt, mirror_opt, resi_vec1, resi_vec2);
        if (outfmt_opt==2) printf("%s%s\t%s%s\t%.4f\n",
            xname.substr(dir1_opt.size()).c_str(), chainID_list1[0].c_str(), 
            yname.substr(dir2_opt.size()).c_str(), chainID_list2[0].c_str(),
            sqrt((TM1*TM1+TM2*TM2)/2));

        /* clean up */
        seqM.clear();
        seqxA.clear();
        seqyA.clear();
        delete[]seqx;
        delete[]seqy;
        delete[]secx;
        delete[]secy;
        DeleteArray(&xa,xlen);
        DeleteArray(&ya,ylen);

        vector<vector<vector<double> > >().swap(xa_vec); // structure of complex1
        vector<vector<vector<double> > >().swap(ya_vec); // structure of complex2
        vector<vector<char> >().swap(seqx_vec); // sequence of complex1
        vector<vector<char> >().swap(seqy_vec); // sequence of complex2
        vector<vector<char> >().swap(secx_vec); // secondary structure of complex1
        vector<vector<char> >().swap(secy_vec); // secondary structure of complex2
        mol_vec1.clear();       // molecule type of complex1, RNA if >0
        mol_vec2.clear();       // molecule type of complex2, RNA if >0
        chainID_list1.clear();  // list of chainID1
        chainID_list2.clear();  // list of chainID2
        xlen_vec.clear();       // length of complex1
        ylen_vec.clear();       // length of complex2
        return 0;
    }

    /* declare TM-score tables */
    int chain1_num=xa_vec.size();
    int chain2_num=ya_vec.size();
    vector<string> tmp_str_vec(chain2_num,"");
    double **TMave_mat;
    NewArray(&TMave_mat,chain1_num,chain2_num);
    vector<vector<string> >seqxA_mat(chain1_num,tmp_str_vec);
    vector<vector<string> > seqM_mat(chain1_num,tmp_str_vec);
    vector<vector<string> >seqyA_mat(chain1_num,tmp_str_vec);

    /* trimComplex */
    vector<vector<vector<double> > > ya_trim_vec; // structure of complex2
    vector<vector<char> >seqy_trim_vec; // sequence of complex2
    vector<vector<char> >secy_trim_vec; // secondary structure of complex2
    vector<int> ylen_trim_vec;          // length of complex2
    int Lchain_aa_max1=0;
    int Lchain_na_max1=0;
    for (i=0;i<chain1_num;i++)
    {
        xlen=xlen_vec[i];
        if      (mol_vec1[i]>0  && xlen>Lchain_na_max1) Lchain_na_max1=xlen;
        else if (mol_vec1[i]<=0 && xlen>Lchain_aa_max1) Lchain_aa_max1=xlen;
    }
    int trim_chain_count=trimComplex(ya_trim_vec,seqy_trim_vec,
        secy_trim_vec,ylen_trim_vec,ya_vec,seqy_vec,secy_vec,ylen_vec,
        mol_vec2,Lchain_aa_max1,Lchain_na_max1);
    int    ylen_trim;             // chain length
    double **ya_trim;             // structure of single chain
    char   *seqy_trim;           // for the protein sequence
    char   *secy_trim;           // for the secondary structure
    double **xt;

    /* get all-against-all alignment */
    if (len_aa+len_na>500) fast_opt=true;
    for (i=0;i<chain1_num;i++)
    {
        xlen=xlen_vec[i];
        if (xlen<3)
        {
            for (j=0;j<chain2_num;j++) TMave_mat[i][j]=-1;
            continue;
        }
        seqx = new char[xlen+1];
        secx = new char[xlen+1];
        NewArray(&xa, xlen, 3);
        copy_chain_data(xa_vec[i],seqx_vec[i],secx_vec[i],
            xlen,xa,seqx,secx);

        for (j=0;j<chain2_num;j++)
        {
            if (mol_vec1[i]*mol_vec2[j]<0) //no protein-RNA alignment
            {
                TMave_mat[i][j]=-1;
                continue;
            }

            ylen=ylen_vec[j];
            if (ylen<3)
            {
                TMave_mat[i][j]=-1;
                continue;
            }
            seqy = new char[ylen+1];
            secy = new char[ylen+1];
            NewArray(&ya, ylen, 3);
            copy_chain_data(ya_vec[j],seqy_vec[j],secy_vec[j],
                ylen,ya,seqy,secy);

            /* declare variable specific to this pair of TMalign */
            double t0[3], u0[3][3];
            double TM1, TM2;
            double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
            double d0_0, TM_0;
            double d0A, d0B, d0u, d0a;
            double d0_out=5.0;
            string seqM, seqxA, seqyA;// for output alignment
            double rmsd0 = 0.0;
            int L_ali;                // Aligned length in standard_TMscore
            double Liden=0;
            double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
            int n_ali=0;
            int n_ali8=0;

            int Lnorm_tmp=len_aa;
            if (mol_vec1[i]+mol_vec2[j]>0) Lnorm_tmp=len_na;

            /* entry function for structure alignment */
            if (trim_chain_count && ylen_trim_vec[j]<ylen)
            {
                ylen_trim = ylen_trim_vec[j];
                seqy_trim = new char[ylen_trim+1];
                secy_trim = new char[ylen_trim+1];
                NewArray(&ya_trim, ylen_trim, 3);
                copy_chain_data(ya_trim_vec[j],seqy_trim_vec[j],secy_trim_vec[j],
                    ylen_trim,ya_trim,seqy_trim,secy_trim);
                TMalign_main(xa, ya_trim, seqx, seqy_trim, secx, secy_trim,
                    t0, u0, TM1, TM2, TM3, TM4, TM5,
                    d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                    seqM, seqxA, seqyA,
                    rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                    xlen, ylen_trim, sequence, Lnorm_tmp, d0_scale,
                    0, false, true, false, fast_opt,
                    mol_vec1[i]+mol_vec2[j],TMcut);
                seqxA.clear();
                seqyA.clear();
                delete[]seqy_trim;
                delete[]secy_trim;
                DeleteArray(&ya_trim,ylen_trim);

                NewArray(&xt,xlen,3);
                do_rotation(xa, xt, xlen, t0, u0);
                int *invmap = new int[ylen+1];
                se_main(xt, ya, seqx, seqy, TM1, TM2, TM3, TM4, TM5,
                    d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
                    rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                    xlen, ylen, sequence, Lnorm_tmp, d0_scale,
                    0, false, 2, false, mol_vec1[i]+mol_vec2[j], 1, invmap);
                delete[]invmap;
                
                if (sequence.size()<2) sequence.push_back("");
                if (sequence.size()<2) sequence.push_back("");
                sequence[0]=seqxA;
                sequence[1]=seqyA;
                TMalign_main(xt, ya, seqx, seqy, secx, secy,
                    t0, u0, TM1, TM2, TM3, TM4, TM5,
                    d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                    seqM, seqxA, seqyA,
                    rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                    xlen, ylen, sequence, Lnorm_tmp, d0_scale,
                    2, false, true, false, fast_opt,
                    mol_vec1[i]+mol_vec2[j],TMcut);
                DeleteArray(&xt, xlen);
            }
            else
            {
                TMalign_main(xa, ya, seqx, seqy, secx, secy,
                    t0, u0, TM1, TM2, TM3, TM4, TM5,
                    d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                    seqM, seqxA, seqyA,
                    rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                    xlen, ylen, sequence, Lnorm_tmp, d0_scale,
                    0, false, true, false, fast_opt,
                    mol_vec1[i]+mol_vec2[j],TMcut);
            }
            
            /* store result */
            seqxA_mat[i][j]=seqxA;
            seqyA_mat[i][j]=seqyA;
            TMave_mat[i][j]=TM4*Lnorm_tmp;

            /* clean up */
            seqM.clear();
            seqxA.clear();
            seqyA.clear();

            delete[]seqy;
            delete[]secy;
            DeleteArray(&ya,ylen);
        }

        delete[]seqx;
        delete[]secx;
        DeleteArray(&xa,xlen);
    }
    vector<vector<vector<double> > >().swap(ya_trim_vec);
    vector<vector<char> >().swap(seqy_trim_vec);
    vector<vector<char> >().swap(secy_trim_vec);
    vector<int> ().swap(ylen_trim_vec);

    /* calculate initial chain-chain assignment */
    int *assign1_list; // value is index of assigned chain2
    int *assign2_list; // value is index of assigned chain1
    assign1_list=new int[chain1_num];
    assign2_list=new int[chain2_num];
    enhanced_greedy_search(TMave_mat, assign1_list,
        assign2_list, chain1_num, chain2_num);

    /* final alignment */
    if (outfmt_opt==0) print_version();
    double **ut_mat; // rotation matrices for all-against-all alignment
    NewArray(&ut_mat,chain1_num,4*3);
    int ui,uj;
    vector<string>xname_vec;
    vector<string>yname_vec;
    vector<double>TM_vec;
    for (i=0;i<chain1_num;i++)
    {
        j=assign1_list[i];
        xname_vec.push_back(xname+chainID_list1[i]);
        if (j<0)
        {
            cerr<<"Warning! "<<chainID_list1[i]<<" cannot be alighed"<<endl;
            for (ui=0;ui<3;ui++)
            {
                for (uj=0;uj<4;uj++) ut_mat[i][ui*3+uj]=0;
                ut_mat[i][ui*3+ui]=1;
            }
            yname_vec.push_back(yname);
            continue;
        }
        yname_vec.push_back(yname+chainID_list2[j]);

        xlen =xlen_vec[i];
        seqx = new char[xlen+1];
        secx = new char[xlen+1];
        NewArray(&xa, xlen, 3);
        copy_chain_data(xa_vec[i],seqx_vec[i],secx_vec[i], xlen,xa,seqx,secx);

        ylen =ylen_vec[j];
        seqy = new char[ylen+1];
        secy = new char[ylen+1];
        NewArray(&ya, ylen, 3);
        copy_chain_data(ya_vec[j],seqy_vec[j],secy_vec[j], ylen,ya,seqy,secy);

        /* declare variable specific to this pair of TMalign */
        double t0[3], u0[3][3];
        double TM1, TM2;
        double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
        double d0_0, TM_0;
        double d0A, d0B, d0u, d0a;
        double d0_out=5.0;
        string seqM, seqxA, seqyA;// for output alignment
        double rmsd0 = 0.0;
        int L_ali;                // Aligned length in standard_TMscore
        double Liden=0;
        double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
        int n_ali=0;
        int n_ali8=0;

        int c;
        for (c=0; c<sequence.size(); c++) sequence[c].clear();
        sequence.clear();
        sequence.push_back(seqxA_mat[i][j]);
        sequence.push_back(seqyA_mat[i][j]);
            
        /* entry function for structure alignment */
        TMalign_main(xa, ya, seqx, seqy, secx, secy,
            t0, u0, TM1, TM2, TM3, TM4, TM5,
            d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
            seqM, seqxA, seqyA,
            rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
            xlen, ylen, sequence, Lnorm_ass, d0_scale,
            3, a_opt, u_opt, d_opt, fast_opt,
            mol_vec1[i]+mol_vec2[j]);
        
        for (ui=0;ui<3;ui++) for (uj=0;uj<3;uj++) ut_mat[i][ui*3+uj]=u0[ui][uj];
        for (uj=0;uj<3;uj++) ut_mat[i][9+uj]=t0[uj];

        TM_vec.push_back(TM1);
        TM_vec.push_back(TM2);

        if (outfmt_opt<2) output_results(
            xname.c_str(), yname.c_str(),
            chainID_list1[i], chainID_list2[j],
            xlen, ylen, t0, u0, TM1, TM2, TM3, TM4, TM5,
            rmsd0, d0_out, seqM.c_str(),
            seqxA.c_str(), seqyA.c_str(), Liden,
            n_ali8, L_ali, TM_ali, rmsd_ali, TM_0, d0_0,
            d0A, d0B, Lnorm_ass, d0_scale, d0a, d0u, 
            "", outfmt_opt, ter_opt, false, split_opt, 
            false, "",//o_opt, fname_super+chainID_list1[i], 
            false, a_opt, u_opt, d_opt, mirror_opt,
            resi_vec1, resi_vec2);
        
        /* clean up */
        seqM.clear();
        seqxA.clear();
        seqyA.clear();

        delete[]seqy;
        delete[]secy;
        DeleteArray(&ya,ylen);

        delete[]seqx;
        delete[]secx;
        DeleteArray(&xa,xlen);
    }
    if (outfmt_opt==2)
    {
        double TM=0;
        for (i=0;i<TM_vec.size();i++) TM+=TM_vec[i]*TM_vec[i];
        TM=sqrt(TM/TM_vec.size());
        string query_name=xname;
        string template_name=yname;
        for (i=0;i<chain1_num;i++)
        {
            j=assign1_list[i];
            if (j<0) continue;
            query_name   +=chainID_list1[i];
            template_name+=chainID_list2[j];
        }
        printf("%s\t%s\t%.4f\n",query_name.c_str(),template_name.c_str(),TM);
        query_name.clear();
        template_name.clear();
    }

    if (m_opt) output_dock_rotation_matrix(fname_matrix.c_str(),
        xname_vec,yname_vec, ut_mat, assign1_list);

    if (o_opt) output_dock(chain1_list, ter_opt, split_opt, infmt1_opt,
        atom_opt, mirror_opt, ut_mat, fname_super);


    /* clean up everything */
    vector<double>().swap(TM_vec);
    vector<string>().swap(xname_vec);
    vector<string>().swap(yname_vec);
    delete [] assign1_list;
    delete [] assign2_list;
    DeleteArray(&TMave_mat,chain1_num);
    DeleteArray(&ut_mat,   chain1_num);
    vector<vector<string> >().swap(seqxA_mat);
    vector<vector<string> >().swap(seqM_mat);
    vector<vector<string> >().swap(seqyA_mat);
    vector<string>().swap(tmp_str_vec);

    vector<vector<vector<double> > >().swap(xa_vec); // structure of complex1
    vector<vector<vector<double> > >().swap(ya_vec); // structure of complex2
    vector<vector<char> >().swap(seqx_vec); // sequence of complex1
    vector<vector<char> >().swap(seqy_vec); // sequence of complex2
    vector<vector<char> >().swap(secx_vec); // secondary structure of complex1
    vector<vector<char> >().swap(secy_vec); // secondary structure of complex2
    mol_vec1.clear();       // molecule type of complex1, RNA if >0
    mol_vec2.clear();       // molecule type of complex2, RNA if >0
    vector<string>().swap(chainID_list1);  // list of chainID1
    vector<string>().swap(chainID_list2);  // list of chainID2
    xlen_vec.clear();       // length of complex1
    ylen_vec.clear();       // length of complex2
    return 1;
}

int mTMalign(string &xname, string &yname, const string &fname_super,
    const string &fname_matrix,
    vector<string> &sequence, double Lnorm_ass, const double d0_scale,
    const bool m_opt, const int  i_opt, const int o_opt, const int a_opt,
    bool u_opt, const bool d_opt, const bool full_opt, const double TMcut,
    const int infmt_opt, const int ter_opt,
    const int split_opt, const int outfmt_opt, bool fast_opt,
    const int het_opt,
    const string &atom_opt, const string &mol_opt, const string &dir_opt,
    const int byresi_opt,
    const vector<string> &chain_list)
{
    /* declare previously global variables */
    vector<vector<vector<double> > >a_vec;  // atomic structure
    vector<vector<vector<double> > >ua_vec; // unchanged atomic structure 
    vector<vector<char> >seq_vec;  // sequence of complex
    vector<vector<char> >sec_vec;  // secondary structure of complex
    vector<int> mol_vec;           // molecule type of complex1, RNA if >0
    vector<string> chainID_list;   // list of chainID
    vector<int> len_vec;           // length of complex
    int    i,j;                    // chain index
    int    xlen, ylen;             // chain length
    double **xa, **ya;             // structure of single chain
    char   *seqx, *seqy;           // for the protein sequence 
    char   *secx, *secy;           // for the secondary structure 
    int    len_aa,len_na;          // total length of protein and RNA/DNA
    vector<string> resi_vec;       // residue index for chain

    /* parse chain list */
    parse_chain_list(chain_list, a_vec, seq_vec, sec_vec, mol_vec,
        len_vec, chainID_list, ter_opt, split_opt, mol_opt, infmt_opt,
        atom_opt, false, het_opt, len_aa, len_na, o_opt, resi_vec);
    int chain_num=a_vec.size();
    if (chain_num<=1) PrintErrorAndQuit("ERROR! <2 chains for multiple alignment");
    if (m_opt||o_opt) for (i=0;i<chain_num;i++) ua_vec.push_back(a_vec[i]);
    int mol_type=0;
    int total_len=0;
    xlen=0;
    for (i=0; i<chain_num; i++)
    {
        if (len_vec[i]>xlen) xlen=len_vec[i];
        total_len+=len_vec[i];
        mol_type+=mol_vec[i];
    }
    if (!u_opt) Lnorm_ass=total_len/chain_num;
    u_opt=true;
    total_len-=xlen;
    if (total_len>750) fast_opt=true;

    /* get all-against-all alignment */
    double **TMave_mat;
    NewArray(&TMave_mat,chain_num,chain_num);
    vector<string> tmp_str_vec(chain_num,"");
    vector<vector<string> >seqxA_mat(chain_num,tmp_str_vec);
    vector<vector<string> >seqyA_mat(chain_num,tmp_str_vec);
    for (i=0;i<chain_num;i++) for (j=0;j<chain_num;j++) TMave_mat[i][j]=0;
    for (i=0;i<chain_num;i++)
    {
        xlen=len_vec[i];
        if (xlen<3) continue;
        seqx = new char[xlen+1];
        secx = new char[xlen+1];
        NewArray(&xa, xlen, 3);
        copy_chain_data(a_vec[i],seq_vec[i],sec_vec[i],xlen,xa,seqx,secx);
        seqxA_mat[i][i]=seqyA_mat[i][i]=(string)(seqx);
        for (j=i+1;j<chain_num;j++)
        {
            ylen=len_vec[j];
            if (ylen<3) continue;
            seqy = new char[ylen+1];
            secy = new char[ylen+1];
            NewArray(&ya, ylen, 3);
            copy_chain_data(a_vec[j],seq_vec[j],sec_vec[j],ylen,ya,seqy,secy);
            
            /* declare variable specific to this pair of TMalign */
            double t0[3], u0[3][3];
            double TM1, TM2;
            double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
            double d0_0, TM_0;
            double d0A, d0B, d0u, d0a;
            double d0_out=5.0;
            string seqM, seqxA, seqyA;// for output alignment
            double rmsd0 = 0.0;
            int L_ali;                // Aligned length in standard_TMscore
            double Liden=0;
            double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
            int n_ali=0;
            int n_ali8=0;

            /* entry function for structure alignment */
            TMalign_main(xa, ya, seqx, seqy, secx, secy,
                t0, u0, TM1, TM2, TM3, TM4, TM5,
                d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                seqM, seqxA, seqyA,
                rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                xlen, ylen, sequence, Lnorm_ass, d0_scale,
                0, false, u_opt, false, fast_opt,
                mol_type,TMcut);

            /* store result */
            TMave_mat[i][j]=TMave_mat[j][i]=TM4;
            seqxA_mat[i][j]=seqyA_mat[j][i]=seqxA;
            seqyA_mat[i][j]=seqxA_mat[j][i]=seqyA;
            //cout<<chain_list[i]<<':'<<chainID_list[i]
                //<<chain_list[j]<<':'<<chainID_list[j]<<"\tTM4="<<TM4<<endl;
            if (full_opt) output_results(
                chain_list[i],chain_list[j], chainID_list[i], chainID_list[j],
                xlen, ylen, t0, u0, TM1, TM2, TM3, TM4, TM5, rmsd0, d0_out,
                seqM.c_str(), seqxA.c_str(), seqyA.c_str(), Liden,
                n_ali8, L_ali, TM_ali, rmsd_ali, TM_0, d0_0, d0A, d0B,
                Lnorm_ass, d0_scale, d0a, d0u, "",
                outfmt_opt, ter_opt, true, split_opt, o_opt, "",
                0, a_opt, false, d_opt, false, resi_vec, resi_vec);

            /* clean up */
            seqM.clear();
            seqxA.clear();
            seqyA.clear();

            delete[]seqy;
            delete[]secy;
            DeleteArray(&ya,ylen);
        }

        delete[]seqx;
        delete[]secx;
        DeleteArray(&xa,xlen);
    }

    /* representative related variables */   
    int r;
    int repr_idx=0;
    vector<string>xname_vec;
    for (i=0;i<chain_num;i++) xname_vec.push_back(
        chain_list[i].substr(dir_opt.size())+chainID_list[i]);
    vector<string>yname_vec;
    double *TMave_list;
    TMave_list = new double[chain_num];
    int *assign_list;
    assign_list=new int[chain_num];
    vector<string> msa(ylen,""); // row is position along msa; column is sequence

    int compare_num;
    double TM1_total, TM2_total;
    double TM3_total, TM4_total, TM5_total;     // for a_opt, u_opt, d_opt
    double d0_0_total, TM_0_total;
    double d0A_total, d0B_total, d0u_total, d0a_total;
    double d0_out_total;
    double rmsd0_total;
    int L_ali_total;                // Aligned length in standard_TMscore
    double Liden_total;
    double TM_ali_total, rmsd_ali_total;  // TMscore and rmsd in standard_TMscore
    int n_ali_total;
    int n_ali8_total;
    int xlen_total, ylen_total;
    double TM4_total_max=0;

    int max_iter=5-(int)(total_len/200);
    if (max_iter<2) max_iter=2;
    int iter=0;
    vector<double> TM_vec(chain_num,0);
    vector<double> d0_vec(chain_num,0);
    vector<double> seqID_vec(chain_num,0);
    vector<vector<double> > TM_mat(chain_num,TM_vec);
    vector<vector<double> > d0_mat(chain_num,d0_vec);
    vector<vector<double> > seqID_mat(chain_num,seqID_vec);
    for (iter=0; iter<max_iter; iter++)
    {
        /* select representative */   
        for (j=0; j<chain_num; j++) TMave_list[j]=0;
        for (i=0; i<chain_num; i++ )
        {
            for (j=0; j<chain_num; j++)
            {
                //cout<<'\t'<<setprecision(4)<<TMave_mat[i][j];
                TMave_list[j]+=TMave_mat[i][j];
            }
            //cout<<'\t'<<chain_list[i]<<':'<<chainID_list[i]<<endl;
        }
        repr_idx=0;
        double repr_TM=0;
        for (j=0; j<chain_num; j++)
        {
            //cout<<chain_list[j]<<'\t'<<len_vec[j]<<'\t'<<TMave_list[j]<<endl;
            if (TMave_list[j]<repr_TM) continue;
            repr_TM=TMave_list[j];
            repr_idx=j;
        }
        //cout<<"repr="<<repr_idx<<"; "<<chain_list[repr_idx]<<"; TM="<<repr_TM<<endl;

        /* superpose superpose */
        yname=chain_list[repr_idx].substr(dir_opt.size())+chainID_list[repr_idx];
        double **xt;
        vector<pair<double,int> >TM_pair_vec; // TM vs chain

        for (i=0; i<chain_num; i++) assign_list[i]=-1;
        assign_list[repr_idx]=repr_idx;
        //ylen = len_vec[repr_idx];
        //seqy = new char[ylen+1];
        //secy = new char[ylen+1];
        //NewArray(&ya, ylen, 3);
        //copy_chain_data(a_vec[repr_idx],seq_vec[repr_idx],sec_vec[repr_idx], ylen,ya,seqy,secy);
        for (r=0;r<sequence.size();r++) sequence[r].clear(); sequence.clear();
        sequence.push_back("");
        sequence.push_back("");
        for (i=0;i<chain_num;i++)
        {
            yname_vec.push_back(yname);
            xlen = len_vec[i];
            if (i==repr_idx || xlen<3) continue;
            TM_pair_vec.push_back(make_pair(-TMave_mat[i][repr_idx],i));
        }
        sort(TM_pair_vec.begin(),TM_pair_vec.end());
    
        int tm_idx;
        if (outfmt_opt<0) cout<<"#PDBchain1\tPDBchain2\tTM1\tTM2\t"
                               <<"RMSD\tID1\tID2\tIDali\tL1\tL2\tLali"<<endl;
        for (tm_idx=0; tm_idx<TM_pair_vec.size(); tm_idx++)
        {
            i=TM_pair_vec[tm_idx].second;
            xlen = len_vec[i];
            seqx = new char[xlen+1];
            secx = new char[xlen+1];
            NewArray(&xa, xlen, 3);
            copy_chain_data(a_vec[i],seq_vec[i],sec_vec[i], xlen,xa,seqx,secx);

            double maxTM=TMave_mat[i][repr_idx];
            int maxj=repr_idx;
            for (j=0;j<chain_num;j++)
            {
                if (i==j || assign_list[j]<0 || TMave_mat[i][j]<=maxTM) continue;
                maxj=j;
                maxTM=TMave_mat[i][j];
            }
            j=maxj;
            assign_list[i]=j;
            ylen = len_vec[j];
            seqy = new char[ylen+1];
            secy = new char[ylen+1];
            NewArray(&ya, ylen, 3);
            copy_chain_data(a_vec[j],seq_vec[j],sec_vec[j], ylen,ya,seqy,secy);

            sequence[0]=seqxA_mat[i][j];
            sequence[1]=seqyA_mat[i][j];
            //cout<<"tm_idx="<<tm_idx<<"\ti="<<i<<"\tj="<<j<<endl;
            //cout<<"superpose "<<xname_vec[i]<<" to "<<xname_vec[j]<<endl;

            /* declare variable specific to this pair of TMalign */
            double t0[3], u0[3][3];
            double TM1, TM2;
            double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
            double d0_0, TM_0;
            double d0A, d0B, d0u, d0a;
            double d0_out=5.0;
            string seqM, seqxA, seqyA;// for output alignment
            double rmsd0 = 0.0;
            int L_ali;                // Aligned length in standard_TMscore
            double Liden=0;
            double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
            int n_ali=0;
            int n_ali8=0;

            /* entry function for structure alignment */
            TMalign_main(xa, ya, seqx, seqy, secx, secy,
                t0, u0, TM1, TM2, TM3, TM4, TM5,
                d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                seqM, seqxA, seqyA,
                rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                xlen, ylen, sequence, Lnorm_ass, d0_scale,
                2,  a_opt, u_opt, d_opt, fast_opt, mol_type);
        
            if (outfmt_opt<0) output_results(
                xname_vec[i].c_str(), xname_vec[j].c_str(), "", "",
                xlen, ylen, t0, u0, TM1, TM2, TM3, TM4, TM5,
                rmsd0, d0_out, seqM.c_str(),
                seqxA.c_str(), seqyA.c_str(), Liden,
                n_ali8, L_ali, TM_ali, rmsd_ali, TM_0, d0_0,
                d0A, d0B, Lnorm_ass, d0_scale, d0a, d0u, 
                "", 2,//outfmt_opt,
                ter_opt, false, split_opt, 
                false, "",//o_opt, fname_super+chainID_list1[i], 
                false, a_opt, u_opt, d_opt, false,
                resi_vec, resi_vec);
         
            NewArray(&xt,xlen,3);
            do_rotation(xa, xt, xlen, t0, u0);
            for (r=0;r<xlen;r++)
            {
                a_vec[i][r][0]=xt[r][0];
                a_vec[i][r][1]=xt[r][1];
                a_vec[i][r][2]=xt[r][2];
            }
            DeleteArray(&xt, xlen);
        
            /* clean up */
            seqM.clear();
            seqxA.clear();
            seqyA.clear();
            sequence[0].clear();
            sequence[1].clear();

            delete[]seqx;
            delete[]secx;
            DeleteArray(&xa,xlen);
        
            delete[]seqy;
            delete[]secy;
            DeleteArray(&ya,ylen);
        }
        ylen = len_vec[repr_idx];
        seqy = new char[ylen+1];
        secy = new char[ylen+1];
        NewArray(&ya, ylen, 3);
        copy_chain_data(a_vec[repr_idx],seq_vec[repr_idx],sec_vec[repr_idx], ylen,ya,seqy,secy);

        /* recover alignment */ 
        int    ylen_ext=ylen;        // chain length
        double **ya_ext;             // structure of single chain
        char   *seqy_ext;            // for the protein sequence 
        char   *secy_ext;            // for the secondary structure 
        for (r=0;r<msa.size();r++) msa[r].clear(); msa.clear();
        msa.assign(ylen,""); // row is position along msa; column is sequence
        vector<string> msa_ext;      // row is position along msa; column is sequence
        for (r=0;r<ylen;r++) msa[r]=seqy[r];
        //for (r=0;r<msa.size();r++) cout<<"["<<r<<"]\t"<<msa[r]<<endl;
        //cout<<"start recover"<<endl;
        assign_list[repr_idx]=0;
        for (tm_idx=0; tm_idx<TM_pair_vec.size(); tm_idx++)
        {
            i=TM_pair_vec[tm_idx].second;
            assign_list[i]=tm_idx+1;

            xlen = len_vec[i];
            seqx = new char[xlen+1];
            secx = new char[xlen+1];
            NewArray(&xa, xlen, 3);
            copy_chain_data(a_vec[i],seq_vec[i],sec_vec[i], xlen,xa,seqx,secx);
        
            /* declare variable specific to this pair of TMalign */
            double TM1, TM2;
            double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
            double d0_0, TM_0;
            double d0A, d0B, d0u, d0a;
            double d0_out=5.0;
            string seqM, seqxA, seqyA;// for output alignment
            double rmsd0 = 0.0;
            int L_ali;                // Aligned length in standard_TMscore
            double Liden=0;
            double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
            int n_ali=0;
            int n_ali8=0;
            int *invmap = new int[ylen+1];

            se_main(xa, ya, seqx, seqy, TM1, TM2, TM3, TM4, TM5,
                d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
                rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                xlen, ylen, sequence, Lnorm_ass, d0_scale,
                0, a_opt, u_opt, d_opt, mol_type, 1, invmap);

            int rx=0,ry=0;
            ylen_ext=seqxA.size();
            NewArray(&ya_ext, ylen_ext, 3);             // structure of single chain
            seqy_ext= new char[ylen_ext+1];            // for the protein sequence 
            secy_ext= new char[ylen_ext+1];            // for the secondary structure 
            string tmp_gap="";
            for (r=0;r<msa[0].size();r++) tmp_gap+='-';
            for (r=msa_ext.size();r<ylen_ext;r++) msa_ext.push_back("");
            //cout<<"x:"<<xname_vec[i]<<'\n'<<seqxA<<endl;
            //cout<<"y:"<<xname_vec[repr_idx]<<'\n'<<seqyA<<endl;
            for (r=0;r<ylen_ext;r++)
            {
                if (seqyA[r]=='-')
                {
                    msa_ext[r]=tmp_gap+seqxA[r];
                    ya_ext[r][0]=xa[rx][0];
                    ya_ext[r][1]=xa[rx][1];
                    ya_ext[r][2]=xa[rx][2];
                    seqy_ext[r]=seqx[rx];
                    secy_ext[r]=secx[rx];
                }
                else
                {
                    msa_ext[r]=msa[ry]+seqxA[r];
                    ya_ext[r][0]=ya[ry][0];
                    ya_ext[r][1]=ya[ry][1];
                    ya_ext[r][2]=ya[ry][2];
                    seqy_ext[r]=seqy[ry];
                    secy_ext[r]=secy[ry];
                }
                rx+=(seqxA[r]!='-');
                ry+=(seqyA[r]!='-');
            }

            /* copy ya_ext to ya */
            delete[]seqy;
            delete[]secy;
            DeleteArray(&ya,ylen);

            ylen=ylen_ext;
            NewArray(&ya,ylen,3);
            seqy = new char[ylen+1];
            secy = new char[ylen+1];
            for (r=0;r<ylen;r++)
            {
                ya[r][0]=ya_ext[r][0];
                ya[r][1]=ya_ext[r][1];
                ya[r][2]=ya_ext[r][2];
                seqy[r]=seqy_ext[r];
                secy[r]=secy_ext[r];
            }
            for (r=0;r<ylen;r++)
            {
                if (r<msa.size()) msa[r]=msa_ext[r];
                else msa.push_back(msa_ext[r]);
            }
            //for (r=0;r<ylen_ext;r++) cout<<"["<<r<<"]\t"<<msa_ext[r]<<'\t'<<seqy[r]<<'\t'
                    //<<ya[r][0]<<'\t'<<ya[r][1]<<'\t'<<ya[r][2]<<'\t'<<secy[r]<<endl;

            /* clean up */
            tmp_gap.clear();
            delete[]invmap;
            seqM.clear();
            seqxA.clear();
            seqyA.clear();

            delete[]seqx;
            delete[]secx;
            DeleteArray(&xa,xlen);

            delete[]seqy_ext;
            delete[]secy_ext;
            DeleteArray(&ya_ext,ylen_ext);
        }
        vector<string>().swap(msa_ext);
        vector<pair<double,int> >().swap(TM_pair_vec);
        for (i=0; i<chain_num; i++)
        {
            tm_idx=assign_list[i];
            if (tm_idx<0) continue;
            seqyA_mat[i][i]="";
            for (r=0 ;r<ylen ; r++) seqyA_mat[i][i]+=msa[r][tm_idx];
            seqxA_mat[i][i]=seqyA_mat[i][i];
            //cout<<xname_vec[i]<<'\t'<<seqxA_mat[i][i]<<endl;
        }
        for (i=0;i<chain_num; i++)
        {
            if (assign_list[i]<0) continue;
            string seqxA=seqxA_mat[i][i];
            for (j=0; j<chain_num; j++)
            {
                if (i==j || assign_list[j]<0) continue;
                string seqyA=seqyA_mat[j][j];
                seqxA_mat[i][j]=seqyA_mat[i][j]="";
                for (r=0;r<ylen;r++)
                {
                    if (seqxA[r]=='-' && seqyA[r]=='-') continue;
                    seqxA_mat[i][j]+=seqxA[r];
                    seqyA_mat[i][j]+=seqyA[r];
                }
                seqyA.clear();
            }
            seqxA.clear();
        }

        /* recover statistics such as TM-score */ 
        compare_num=0;
        TM1_total=0, TM2_total=0;
        TM3_total=0, TM4_total=0, TM5_total=0;
        d0_0_total=0, TM_0_total=0;
        d0A_total=0, d0B_total=0, d0u_total=0, d0a_total=0;
        d0_out_total=0;
        rmsd0_total = 0.0;
        L_ali_total=0;
        Liden_total=0;
        TM_ali_total=0, rmsd_ali_total=0;
        n_ali_total=0;
        n_ali8_total=0;
        xlen_total=0, ylen_total=0;
        for (i=0; i< chain_num; i++)
        {
            xlen=len_vec[i];
            if (xlen<3) continue;
            seqx = new char[xlen+1];
            secx = new char[xlen+1];
            NewArray(&xa, xlen, 3);
            copy_chain_data(a_vec[i],seq_vec[i],sec_vec[i], xlen,xa,seqx,secx);
            for (j=i+1;j<chain_num;j++)
            {
                ylen=len_vec[j];
                if (ylen<3) continue;
                compare_num++;
                seqy = new char[ylen+1];
                secy = new char[ylen+1];
                NewArray(&ya, ylen, 3);
                copy_chain_data(a_vec[j],seq_vec[j],sec_vec[j],ylen,ya,seqy,secy);
                sequence[0]=seqxA_mat[i][j];
                sequence[1]=seqyA_mat[i][j];
            
                /* declare variable specific to this pair of TMalign */
                double TM1, TM2;
                double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
                double d0_0, TM_0;
                double d0A, d0B, d0u, d0a;
                double d0_out=5.0;
                string seqM, seqxA, seqyA;// for output alignment
                double rmsd0 = 0.0;
                int L_ali=0;              // Aligned length in standard_TMscore
                double Liden=0;
                double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
                int n_ali=0;
                int n_ali8=0;
                int *invmap = new int[ylen+1];

                se_main(xa, ya, seqx, seqy, TM1, TM2, TM3, TM4, TM5,
                    d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out, seqM, seqxA, seqyA,
                    rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                    xlen, ylen, sequence, Lnorm_ass, d0_scale,
                    true, a_opt, u_opt, d_opt, mol_type, 1, invmap);

                if (xlen<=ylen)
                {
                    xlen_total+=xlen;
                    ylen_total+=ylen;
                    TM1_total+=TM1;
                    TM2_total+=TM2;
                    d0A_total+=d0A;
                    d0B_total+=d0B;
                }
                else
                {
                    xlen_total+=ylen;
                    ylen_total+=xlen;
                    TM1_total+=TM2;
                    TM2_total+=TM1;
                    d0A_total+=d0B;
                    d0B_total+=d0A;
                }
                TM_mat[i][j]=TM2;
                TM_mat[j][i]=TM1;
                d0_mat[i][j]=d0B;
                d0_mat[j][i]=d0A;
                seqID_mat[i][j]=1.*Liden/xlen;
                seqID_mat[j][i]=1.*Liden/ylen;

                TM3_total+=TM3;
                TM4_total+=TM4;
                TM5_total+=TM5;
                d0_0_total+=d0_0;
                TM_0_total+=TM_0;
                d0u_total+=d0u;
                d0_out_total+=d0_out;
                rmsd0_total+=rmsd0;
                L_ali_total+=L_ali;        // Aligned length in standard_TMscore
                Liden_total+=Liden;
                TM_ali_total+=TM_ali;
                rmsd_ali_total+=rmsd_ali;  // TMscore and rmsd in standard_TMscore
                n_ali_total+=n_ali;
                n_ali8_total+=n_ali8;

                /* clean up */
                delete[]invmap;
                seqM.clear();
                seqxA.clear();
                seqyA.clear();

                delete[]seqy;
                delete[]secy;
                DeleteArray(&ya,ylen);
            }
            delete[]seqx;
            delete[]secx;
            DeleteArray(&xa,xlen);
        }
        if (TM4_total<=TM4_total_max) break;
        TM4_total_max=TM4_total;
    }
    for (i=0;i<chain_num;i++)
    {
        for (j=0;j<chain_num;j++)
        {
            if (i==j) continue;
            TM_vec[i]+=TM_mat[i][j];
            d0_vec[i]+=d0_mat[i][j];
            seqID_vec[i]+=seqID_mat[i][j];
        }
        TM_vec[i]/=(chain_num-1);
        d0_vec[i]/=(chain_num-1);
        seqID_vec[i]/=(chain_num-1);
    }
    xlen_total    /=compare_num;
    ylen_total    /=compare_num;
    TM1_total     /=compare_num;
    TM2_total     /=compare_num;
    d0A_total     /=compare_num;
    d0B_total     /=compare_num;
    TM3_total     /=compare_num;
    TM4_total     /=compare_num;
    TM5_total     /=compare_num;
    d0_0_total    /=compare_num;
    TM_0_total    /=compare_num;
    d0u_total     /=compare_num;
    d0_out_total  /=compare_num;
    rmsd0_total   /=compare_num;
    L_ali_total   /=compare_num;
    Liden_total   /=compare_num;
    TM_ali_total  /=compare_num;
    rmsd_ali_total/=compare_num;
    n_ali_total   /=compare_num;
    n_ali8_total  /=compare_num;
    xname="shorter";
    yname="longer";
    string seqM="";
    string seqxA="";
    string seqyA="";
    double t0[3];
    double u0[3][3];
    stringstream buf;
    for (i=0; i<chain_num; i++)
    {
        if (assign_list[i]<0) continue;
        buf <<">"<<xname_vec[i]<<"\tL="<<len_vec[i]
            <<"\td0="<<setiosflags(ios::fixed)<<setprecision(2)<<d0_vec[i]
            <<"\tseqID="<<setiosflags(ios::fixed)<<setprecision(3)<<seqID_vec[i]
            <<"\tTM-score="<<setiosflags(ios::fixed)<<setprecision(5)<<TM_vec[i];
        if (i==repr_idx) buf<<"\t*";
        buf<<'\n'<<seqxA_mat[i][i]<<endl;
    }
    seqM=buf.str();
    seqM=seqM.substr(0,seqM.size()-1);
    buf.str(string());
    //MergeAlign(seqxA_mat,seqyA_mat,repr_idx,xname_vec,chain_num,seqM);
    if (outfmt_opt==0) print_version();
    output_mTMalign_results( xname,yname, "","",
        xlen_total, ylen_total, t0, u0, TM1_total, TM2_total, 
        TM3_total, TM4_total, TM5_total, rmsd0_total, d0_out_total,
        seqM.c_str(), seqxA.c_str(), seqyA.c_str(), Liden_total,
        n_ali8_total, L_ali_total, TM_ali_total, rmsd_ali_total,
        TM_0_total, d0_0_total, d0A_total, d0B_total,
        Lnorm_ass, d0_scale, d0a_total, d0u_total, 
        "", outfmt_opt, ter_opt, 0, split_opt, false,
        "", false, a_opt, u_opt, d_opt, false,
        resi_vec, resi_vec );

    if (m_opt || o_opt)
    {
        double **ut_mat; // rotation matrices for all-against-all alignment
        int ui,uj;
        double t[3], u[3][3];
        double rmsd;
        NewArray(&ut_mat,chain_num,4*3);
        for (i=0;i<chain_num;i++)
        {
            xlen=ylen=a_vec[i].size();
            NewArray(&xa,xlen,3);
            NewArray(&ya,xlen,3);
            for (r=0;r<xlen;r++)
            {
                xa[r][0]=ua_vec[i][r][0];
                xa[r][1]=ua_vec[i][r][1];
                xa[r][2]=ua_vec[i][r][2];
                ya[r][0]= a_vec[i][r][0];
                ya[r][1]= a_vec[i][r][1];
                ya[r][2]= a_vec[i][r][2];
            }
            Kabsch(xa,ya,xlen,1,&rmsd,t,u);
            for (ui=0;ui<3;ui++) for (uj=0;uj<3;uj++) ut_mat[i][ui*3+uj]=u[ui][uj];
            for (uj=0;uj<3;uj++) ut_mat[i][9+uj]=t[uj];
            DeleteArray(&xa,xlen);
            DeleteArray(&ya,xlen);
        }
        vector<vector<vector<double> > >().swap(ua_vec);

        if (m_opt)
        {
            assign_list[repr_idx]=-1;
            output_dock_rotation_matrix(fname_matrix.c_str(),
                xname_vec,yname_vec, ut_mat, assign_list);
        }

        if (o_opt) output_dock(chain_list, ter_opt, split_opt, 
                infmt_opt, atom_opt, false, ut_mat, fname_super);
        
        DeleteArray(&ut_mat,chain_num);
    }

    /* clean up */
    vector<string>().swap(msa);
    vector<string>().swap(tmp_str_vec);
    vector<vector<string> >().swap(seqxA_mat);
    vector<vector<string> >().swap(seqyA_mat);
    vector<string>().swap(xname_vec);
    vector<string>().swap(yname_vec);
    delete[]TMave_list;
    DeleteArray(&TMave_mat,chain_num);
    vector<vector<vector<double> > >().swap(a_vec); // structure of complex
    vector<vector<char> >().swap(seq_vec); // sequence of complex
    vector<vector<char> >().swap(sec_vec); // secondary structure of complex
    vector<int>().swap(mol_vec);           // molecule type of complex1, RNA if >0
    vector<string>().swap(chainID_list);   // list of chainID
    vector<int>().swap(len_vec);           // length of complex
    vector<double>().swap(TM_vec);
    vector<double>().swap(d0_vec);
    vector<double>().swap(seqID_vec);
    vector<vector<double> >().swap(TM_mat);
    vector<vector<double> >().swap(d0_mat);
    vector<vector<double> >().swap(seqID_mat);
    return 1;
}

/* sequence order independent alignment */
int SOIalign(string &xname, string &yname, const string &fname_super,
    const string &fname_lign, const string &fname_matrix,
    vector<string> &sequence, const double Lnorm_ass, const double d0_scale,
    const bool m_opt, const int  i_opt, const int o_opt, const int a_opt,
    const bool u_opt, const bool d_opt, const double TMcut,
    const int infmt1_opt, const int infmt2_opt, const int ter_opt,
    const int split_opt, const int outfmt_opt, const bool fast_opt,
    const int cp_opt, const int mirror_opt, const int het_opt,
    const string &atom_opt, const string &mol_opt, const string &dir_opt,
    const string &dir1_opt, const string &dir2_opt, 
    const vector<string> &chain1_list, const vector<string> &chain2_list,
    const bool se_opt, const int closeK_opt, const int mm_opt)
{
    /* declare previously global variables */
    vector<vector<string> >PDB_lines1; // text of chain1
    vector<vector<string> >PDB_lines2; // text of chain2
    vector<int> mol_vec1;              // molecule type of chain1, RNA if >0
    vector<int> mol_vec2;              // molecule type of chain2, RNA if >0
    vector<string> chainID_list1;      // list of chainID1
    vector<string> chainID_list2;      // list of chainID2
    int    i,j;                // file index
    int    chain_i,chain_j;    // chain index
    int    r;                  // residue index
    int    xlen, ylen;         // chain length
    int    xchainnum,ychainnum;// number of chains in a PDB file
    char   *seqx, *seqy;       // for the protein sequence 
    char   *secx, *secy;       // for the secondary structure 
    int    **secx_bond;        // boundary of secondary structure
    int    **secy_bond;        // boundary of secondary structure
    double **xa, **ya;         // for input vectors xa[0...xlen-1][0..2] and
                               // ya[0...ylen-1][0..2], in general,
                               // ya is regarded as native structure 
                               // --> superpose xa onto ya
    double **xk, **yk;         // k closest residues
    vector<string> resi_vec1;  // residue index for chain1
    vector<string> resi_vec2;  // residue index for chain2
    int read_resi=0;  // whether to read residue index
    if (o_opt) read_resi=2;

    /* loop over file names */
    for (i=0;i<chain1_list.size();i++)
    {
        /* parse chain 1 */
        xname=chain1_list[i];
        xchainnum=get_PDB_lines(xname, PDB_lines1, chainID_list1,
            mol_vec1, ter_opt, infmt1_opt, atom_opt, split_opt, het_opt);
        if (!xchainnum)
        {
            cerr<<"Warning! Cannot parse file: "<<xname
                <<". Chain number 0."<<endl;
            continue;
        }
        for (chain_i=0;chain_i<xchainnum;chain_i++)
        {
            xlen=PDB_lines1[chain_i].size();
            if (mol_opt=="RNA") mol_vec1[chain_i]=1;
            else if (mol_opt=="protein") mol_vec1[chain_i]=-1;
            if (!xlen)
            {
                cerr<<"Warning! Cannot parse file: "<<xname
                    <<". Chain length 0."<<endl;
                continue;
            }
            else if (xlen<3)
            {
                cerr<<"Sequence is too short <3!: "<<xname<<endl;
                continue;
            }
            NewArray(&xa, xlen, 3);
            if (closeK_opt>=3) NewArray(&xk, xlen*closeK_opt, 3);
            seqx = new char[xlen + 1];
            secx = new char[xlen + 1];
            xlen = read_PDB(PDB_lines1[chain_i], xa, seqx, 
                resi_vec1, read_resi);
            if (mirror_opt) for (r=0;r<xlen;r++) xa[r][2]=-xa[r][2];
            if (mol_vec1[chain_i]>0) make_sec(seqx,xa, xlen, secx,atom_opt);
            else make_sec(xa, xlen, secx); // secondary structure assignment
            if (closeK_opt>=3) getCloseK(xa, xlen, closeK_opt, xk);
            if (mm_opt==6) 
            {
                NewArray(&secx_bond, xlen, 2);
                assign_sec_bond(secx_bond, secx, xlen);
            }

            for (j=(dir_opt.size()>0)*(i+1);j<chain2_list.size();j++)
            {
                /* parse chain 2 */
                if (PDB_lines2.size()==0)
                {
                    yname=chain2_list[j];
                    ychainnum=get_PDB_lines(yname, PDB_lines2, chainID_list2,
                        mol_vec2, ter_opt, infmt2_opt, atom_opt, split_opt,
                        het_opt);
                    if (!ychainnum)
                    {
                        cerr<<"Warning! Cannot parse file: "<<yname
                            <<". Chain number 0."<<endl;
                        continue;
                    }
                }
                for (chain_j=0;chain_j<ychainnum;chain_j++)
                {
                    ylen=PDB_lines2[chain_j].size();
                    if (mol_opt=="RNA") mol_vec2[chain_j]=1;
                    else if (mol_opt=="protein") mol_vec2[chain_j]=-1;
                    if (!ylen)
                    {
                        cerr<<"Warning! Cannot parse file: "<<yname
                            <<". Chain length 0."<<endl;
                        continue;
                    }
                    else if (ylen<3)
                    {
                        cerr<<"Sequence is too short <3!: "<<yname<<endl;
                        continue;
                    }
                    NewArray(&ya, ylen, 3);
                    if (closeK_opt>=3) NewArray(&yk, ylen*closeK_opt, 3);
                    seqy = new char[ylen + 1];
                    secy = new char[ylen + 1];
                    ylen = read_PDB(PDB_lines2[chain_j], ya, seqy,
                        resi_vec2, read_resi);
                    if (mol_vec2[chain_j]>0)
                         make_sec(seqy, ya, ylen, secy, atom_opt);
                    else make_sec(ya, ylen, secy);
                    if (closeK_opt>=3) getCloseK(ya, ylen, closeK_opt, yk);
                    if (mm_opt==6) 
                    {
                        NewArray(&secy_bond, ylen, 2);
                        assign_sec_bond(secy_bond, secy, ylen);
                    }

                    /* declare variable specific to this pair of TMalign */
                    double t0[3], u0[3][3];
                    double TM1, TM2;
                    double TM3, TM4, TM5;     // for a_opt, u_opt, d_opt
                    double d0_0, TM_0;
                    double d0A, d0B, d0u, d0a;
                    double d0_out=5.0;
                    string seqM, seqxA, seqyA;// for output alignment
                    double rmsd0 = 0.0;
                    int L_ali;                // Aligned length in standard_TMscore
                    double Liden=0;
                    double TM_ali, rmsd_ali;  // TMscore and rmsd in standard_TMscore
                    int n_ali=0;
                    int n_ali8=0;
                    bool force_fast_opt=(getmin(xlen,ylen)>1500)?true:fast_opt;
                    int *invmap = new int[ylen+1];
                    double *dist_list = new double[ylen+1];

                    /* entry function for structure alignment */
                    if (se_opt) 
                    {
                        u0[0][0]=u0[1][1]=u0[2][2]=1;
                        u0[0][1]=         u0[0][2]=
                        u0[1][0]=         u0[1][2]=
                        u0[2][0]=         u0[2][1]=
                        t0[0]   =t0[1]   =t0[2]   =0;
                        soi_se_main(
                            xa, ya, seqx, seqy, TM1, TM2, TM3, TM4, TM5,
                            d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                            seqM, seqxA, seqyA,
                            rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                            xlen, ylen, Lnorm_ass, d0_scale,
                            i_opt, a_opt, u_opt, d_opt,
                            mol_vec1[chain_i]+mol_vec2[chain_j], 
                            outfmt_opt, invmap, dist_list,
                            secx_bond, secy_bond, mm_opt);
                        if (outfmt_opt>=2) 
                        {
                            Liden=L_ali=0;
                            int r1,r2;
                            for (r2=0;r2<ylen;r2++)
                            {
                                r1=invmap[r2];
                                if (r1<0) continue;
                                L_ali+=1;
                                Liden+=(seqx[r1]==seqy[r2]);
                            }
                        }
                    }
                    else SOIalign_main(xa, ya, xk, yk, closeK_opt,
                        seqx, seqy, secx, secy,
                        t0, u0, TM1, TM2, TM3, TM4, TM5,
                        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
                        seqM, seqxA, seqyA, invmap,
                        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
                        xlen, ylen, sequence, Lnorm_ass, d0_scale,
                        i_opt, a_opt, u_opt, d_opt, force_fast_opt,
                        mol_vec1[chain_i]+mol_vec2[chain_j], dist_list,
                        secx_bond, secy_bond, mm_opt);

                    /* print result */
                    if (outfmt_opt==0) print_version();
                    output_results(
                        xname.substr(dir1_opt.size()+dir_opt.size()),
                        yname.substr(dir2_opt.size()+dir_opt.size()),
                        chainID_list1[chain_i], chainID_list2[chain_j],
                        xlen, ylen, t0, u0, TM1, TM2, TM3, TM4, TM5,
                        rmsd0, d0_out, seqM.c_str(),
                        seqxA.c_str(), seqyA.c_str(), Liden,
                        n_ali8, L_ali, TM_ali, rmsd_ali, TM_0, d0_0,
                        d0A, d0B, Lnorm_ass, d0_scale, d0a, d0u, 
                        (m_opt?fname_matrix:"").c_str(),
                        outfmt_opt, ter_opt, false, split_opt, o_opt,
                        fname_super, i_opt, a_opt, u_opt, d_opt, mirror_opt,
                        resi_vec1, resi_vec2);
                    if (outfmt_opt<=0)
                    {
                        cout<<"###############\t###############\t#########"<<endl;
                        cout<<"#Aligned atom 1\tAligned atom 2 \tDistance#"<<endl;
                        int r1,r2;
                        for (r2=0;r2<ylen;r2++)
                        {
                            r1=invmap[r2];
                            if (r1<0) continue;
                            cout<<PDB_lines1[chain_i][r1].substr(12,15)<<'\t'
                                <<PDB_lines2[chain_j][r2].substr(12,15)<<'\t'
                                <<setw(9)<<setiosflags(ios::fixed)<<setprecision(3)
                                <<dist_list[r2]<<'\n';
                        }
                        cout<<"###############\t###############\t#########"<<endl;
                    }

                    /* Done! Free memory */
                    delete [] invmap;
                    delete [] dist_list;
                    seqM.clear();
                    seqxA.clear();
                    seqyA.clear();
                    DeleteArray(&ya, ylen);
                    if (closeK_opt>=3) DeleteArray(&yk, ylen*closeK_opt);
                    delete [] seqy;
                    delete [] secy;
                    resi_vec2.clear();
                    if (mm_opt==6) DeleteArray(&secy_bond, ylen);
                } // chain_j
                if (chain2_list.size()>1)
                {
                    yname.clear();
                    for (chain_j=0;chain_j<ychainnum;chain_j++)
                        PDB_lines2[chain_j].clear();
                    PDB_lines2.clear();
                    chainID_list2.clear();
                    mol_vec2.clear();
                }
            } // j
            PDB_lines1[chain_i].clear();
            DeleteArray(&xa, xlen);
            if (closeK_opt>=3) DeleteArray(&xk, xlen*closeK_opt);
            delete [] seqx;
            delete [] secx;
            resi_vec1.clear();
            if (mm_opt==6) DeleteArray(&secx_bond, xlen);
        } // chain_i
        xname.clear();
        PDB_lines1.clear();
        chainID_list1.clear();
        mol_vec1.clear();
    } // i
    if (chain2_list.size()==1)
    {
        yname.clear();
        for (chain_j=0;chain_j<ychainnum;chain_j++)
            PDB_lines2[chain_j].clear();
        PDB_lines2.clear();
        resi_vec2.clear();
        chainID_list2.clear();
        mol_vec2.clear();
    }
    return 0;
}


int main(int argc, char *argv[])
{
    if (argc < 2) print_help();


    clock_t t1, t2;
    t1 = clock();

    /**********************/
    /*    get argument    */
    /**********************/
    string xname       = "";
    string yname       = "";
    string fname_super = ""; // file name for superposed structure
    string fname_lign  = ""; // file name for user alignment
    string fname_matrix= ""; // file name for output matrix
    vector<string> sequence; // get value from alignment file
    double Lnorm_ass, d0_scale;

    bool h_opt = false; // print full help message
    bool v_opt = false; // print version
    bool m_opt = false; // flag for -m, output rotation matrix
    int  i_opt = 0;     // 1 for -i, 3 for -I
    int  o_opt = 0;     // 1 for -o, 2 for -rasmol
    int  a_opt = 0;     // flag for -a, do not normalized by average length
    bool u_opt = false; // flag for -u, normalized by user specified length
    bool d_opt = false; // flag for -d, user specified d0

    bool   full_opt  = false;// do not show chain level alignment
    double TMcut     =-1;
    bool   se_opt    =false;
    int    infmt1_opt=-1;    // PDB or PDBx/mmCIF format for chain_1
    int    infmt2_opt=-1;    // PDB or PDBx/mmCIF format for chain_2
    int    ter_opt   =2;     // END, or different chainID
    int    split_opt =2;     // split each chains
    int    outfmt_opt=0;     // set -outfmt to full output
    bool   fast_opt  =false; // flags for -fast, fTM-align algorithm
    int    cp_opt    =0;     // do not check circular permutation
    int    closeK_opt=-1;    // number of atoms for SOI initial alignment.
                             // 5 and 0 for -mm 5 and 6
    int    hinge_opt =9;     // maximum number of hinge allowed for flexible
    int    mirror_opt=0;     // do not align mirror
    int    het_opt=0;        // do not read HETATM residues
    int    mm_opt=0;         // do not perform MM-align
    string atom_opt  ="auto";// use C alpha atom for protein and C3' for RNA
    string mol_opt   ="auto";// auto-detect the molecule type as protein/RNA
    string suffix_opt="";    // set -suffix to empty
    string dir_opt   ="";    // set -dir to empty
    string dir1_opt  ="";    // set -dir1 to empty
    string dir2_opt  ="";    // set -dir2 to empty
    int    byresi_opt=0;     // set -byresi to 0
    vector<string> chain1_list; // only when -dir1 is set
    vector<string> chain2_list; // only when -dir2 is set

    for(int i = 1; i < argc; i++)
    {
        if ( !strcmp(argv[i],"-o") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -o");
            if (o_opt==2)
                cerr<<"Warning! -rasmol is already set. Ignore -o"<<endl;
            else
            {
                fname_super = argv[i + 1];
                o_opt = 1;
            }
            i++;
        }
        else if ( !strcmp(argv[i],"-rasmol") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -rasmol");
            if (o_opt==1)
                cerr<<"Warning! -o is already set. Ignore -rasmol"<<endl;
            else
            {
                fname_super = argv[i + 1];
                o_opt = 2;
            }
            i++;
        }
        else if ( !strcmp(argv[i],"-u") || !strcmp(argv[i],"-L") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -u or -L");
            Lnorm_ass = atof(argv[i + 1]); u_opt = true; i++;
            if (Lnorm_ass<=0) PrintErrorAndQuit(
                "ERROR! The value for -u or -L should be >0");
        }
        else if ( !strcmp(argv[i],"-a") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -a");
            if (!strcmp(argv[i + 1], "T"))      a_opt=true;
            else if (!strcmp(argv[i + 1], "F")) a_opt=false;
            else 
            {
                a_opt=atoi(argv[i + 1]);
                if (a_opt!=-2 && a_opt!=-1 && a_opt!=1)
                    PrintErrorAndQuit("-a must be -2, -1, 1, T or F");
            }
            i++;
        }
        else if ( !strcmp(argv[i],"-full") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -full");
            if (!strcmp(argv[i + 1], "T"))      full_opt=true;
            else if (!strcmp(argv[i + 1], "F")) full_opt=false;
            else PrintErrorAndQuit("-full must be T or F");
            i++;
        }
        else if ( !strcmp(argv[i],"-d") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -d");
            d0_scale = atof(argv[i + 1]); d_opt = true; i++;
        }
        else if ( !strcmp(argv[i],"-closeK") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -closeK");
            closeK_opt = atoi(argv[i + 1]); i++;
        }
        else if ( !strcmp(argv[i],"-hinge") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -hinge");
            hinge_opt = atoi(argv[i + 1]); i++;
        }
        else if ( !strcmp(argv[i],"-v") )
        {
            v_opt = true;
        }
        else if ( !strcmp(argv[i],"-h") )
        {
            h_opt = true;
        }
        else if ( !strcmp(argv[i],"-i") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -i");
            if (i_opt==3)
                PrintErrorAndQuit("ERROR! -i and -I cannot be used together");
            fname_lign = argv[i + 1];      i_opt = 1; i++;
        }
        else if (!strcmp(argv[i], "-I") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -I");
            if (i_opt==1)
                PrintErrorAndQuit("ERROR! -I and -i cannot be used together");
            fname_lign = argv[i + 1];      i_opt = 3; i++;
        }
        else if (!strcmp(argv[i], "-m") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -m");
            fname_matrix = argv[i + 1];    m_opt = true; i++;
        }// get filename for rotation matrix
        else if (!strcmp(argv[i], "-fast"))
        {
            fast_opt = true;
        }
        else if (!strcmp(argv[i], "-se"))
        {
            se_opt = true;
        }
        else if ( !strcmp(argv[i],"-infmt1") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -infmt1");
            infmt1_opt=atoi(argv[i + 1]); i++;
            if (infmt1_opt<-1 || infmt1_opt>3)
                PrintErrorAndQuit("ERROR! -infmt1 can only be -1, 0, 1, 2, or 3");
        }
        else if ( !strcmp(argv[i],"-infmt2") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -infmt2");
            infmt2_opt=atoi(argv[i + 1]); i++;
            if (infmt2_opt<-1 || infmt2_opt>3)
                PrintErrorAndQuit("ERROR! -infmt2 can only be -1, 0, 1, 2, or 3");
        }
        else if ( !strcmp(argv[i],"-ter") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -ter");
            ter_opt=atoi(argv[i + 1]); i++;
        }
        else if ( !strcmp(argv[i],"-split") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -split");
            split_opt=atoi(argv[i + 1]); i++;
        }
        else if ( !strcmp(argv[i],"-atom") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -atom");
            atom_opt=argv[i + 1]; i++;
            if (atom_opt.size()!=4) PrintErrorAndQuit(
                "ERROR! Atom name must have 4 characters, including space.\n"
                "For example, C alpha, C3' and P atoms should be specified by\n"
                "-atom \" CA \", -atom \" P  \" and -atom \" C3'\", respectively.");
        }
        else if ( !strcmp(argv[i],"-mol") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -mol");
            mol_opt=argv[i + 1]; i++;
            if (mol_opt=="prot") mol_opt="protein";
            else if (mol_opt=="DNA") mol_opt="RNA";
            if (mol_opt!="auto" && mol_opt!="protein" && mol_opt!="RNA")
                PrintErrorAndQuit("ERROR! Molecule type must be one of the "
                    "following:\nauto, prot (the same as 'protein'), and "
                    "RNA (the same as 'DNA').");
        }
        else if ( !strcmp(argv[i],"-dir") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -dir");
            dir_opt=argv[i + 1]; i++;
        }
        else if ( !strcmp(argv[i],"-dir1") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -dir1");
            dir1_opt=argv[i + 1]; i++;
        }
        else if ( !strcmp(argv[i],"-dir2") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -dir2");
            dir2_opt=argv[i + 1]; i++;
        }
        else if ( !strcmp(argv[i],"-suffix") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -suffix");
            suffix_opt=argv[i + 1]; i++;
        }
        else if ( !strcmp(argv[i],"-outfmt") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -outfmt");
            outfmt_opt=atoi(argv[i + 1]); i++;
        }
        else if ( !strcmp(argv[i],"-TMcut") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -TMcut");
            TMcut=atof(argv[i + 1]); i++;
        }
        else if ( !strcmp(argv[i],"-byresi")  || 
                  !strcmp(argv[i],"-tmscore") ||
                  !strcmp(argv[i],"-TMscore"))
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -byresi");
            byresi_opt=atoi(argv[i + 1]); i++;
        }
        else if ( !strcmp(argv[i],"-seq") )
        {
            byresi_opt=5;
        }
        else if ( !strcmp(argv[i],"-cp") )
        {
            mm_opt==3;
        }
        else if ( !strcmp(argv[i],"-mirror") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -mirror");
            mirror_opt=atoi(argv[i + 1]); i++;
        }
        else if ( !strcmp(argv[i],"-het") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -het");
            het_opt=atoi(argv[i + 1]); i++;
            if (het_opt!=0 && het_opt!=1 && het_opt!=2)
                PrintErrorAndQuit("-het must be 0, 1, or 2");
        }
        else if ( !strcmp(argv[i],"-mm") )
        {
            if (i>=(argc-1)) 
                PrintErrorAndQuit("ERROR! Missing value for -mm");
            mm_opt=atoi(argv[i + 1]); i++;
        }
        else if (xname.size() == 0) xname=argv[i];
        else if (yname.size() == 0) yname=argv[i];
        else PrintErrorAndQuit(string("ERROR! Undefined option ")+argv[i]);
    }

    if(xname.size()==0 || (yname.size()==0 && dir_opt.size()==0) || 
                          (yname.size()    && dir_opt.size()))
    {
        if (h_opt) print_help(h_opt);
        if (v_opt)
        {
            print_version();
            exit(EXIT_FAILURE);
        }
        if (xname.size()==0)
            PrintErrorAndQuit("Please provide input structures");
        else if (yname.size()==0 && dir_opt.size()==0 && mm_opt!=4)
            PrintErrorAndQuit("Please provide structure B");
        else if (yname.size() && dir_opt.size())
            PrintErrorAndQuit("Please provide only one file name if -dir is set");
    }

    if (suffix_opt.size() && dir_opt.size()+dir1_opt.size()+dir2_opt.size()==0)
        PrintErrorAndQuit("-suffix is only valid if -dir, -dir1 or -dir2 is set");
    if ((dir_opt.size() || dir1_opt.size() || dir2_opt.size()))
    {
        if (mm_opt!=2 && mm_opt!=4 && (m_opt || o_opt))
            PrintErrorAndQuit("-m or -o cannot be set with -dir, -dir1 or -dir2");
        else if (dir_opt.size() && (dir1_opt.size() || dir2_opt.size()))
            PrintErrorAndQuit("-dir cannot be set with -dir1 or -dir2");
    }
    if (o_opt && (infmt1_opt!=-1 && infmt1_opt!=0 && infmt1_opt!=3))
        PrintErrorAndQuit("-o can only be used with -infmt1 -1, 0 or 3");

    if (mol_opt=="protein" && atom_opt=="auto")
        atom_opt=" CA ";
    else if (mol_opt=="RNA" && atom_opt=="auto")
        atom_opt=" C3'";

    if (d_opt && d0_scale<=0)
        PrintErrorAndQuit("Wrong value for option -d! It should be >0");
    if (outfmt_opt>=2 && (a_opt || u_opt || d_opt))
        PrintErrorAndQuit("-outfmt 2 cannot be used with -a, -u, -L, -d");
    if (byresi_opt!=0)
    {
        if (i_opt)
            PrintErrorAndQuit("-byresi >=1 cannot be used with -i or -I");
        if (byresi_opt<0 || byresi_opt>6)
            PrintErrorAndQuit("-byresi can only be 0 to 6");
        if ((byresi_opt==2 || byresi_opt==3 || byresi_opt==6) && ter_opt>=2)
            PrintErrorAndQuit("-byresi 2 and 6 must be used with -ter <=1");
    }
    //if (split_opt==1 && ter_opt!=0)
        //PrintErrorAndQuit("-split 1 should be used with -ter 0");
    //else if (split_opt==2 && ter_opt!=0 && ter_opt!=1)
        //PrintErrorAndQuit("-split 2 should be used with -ter 0 or 1");
    if (split_opt<0 || split_opt>2)
        PrintErrorAndQuit("-split can only be 0, 1 or 2");

    if (mm_opt==3)
    {
        cp_opt=true;
        mm_opt=0;
    }
    if (cp_opt && i_opt)
        PrintErrorAndQuit("-mm 3 cannot be used with -i or -I");

    if (mirror_opt && het_opt!=1)
        cerr<<"WARNING! -mirror was not used with -het 1. "
            <<"D amino acids may not be correctly aligned."<<endl;

    if (mm_opt)
    {
        if (i_opt) PrintErrorAndQuit("-mm cannot be used with -i or -I");
        if (u_opt) PrintErrorAndQuit("-mm cannot be used with -u or -L");
        //if (cp_opt) PrintErrorAndQuit("-mm cannot be used with -cp");
        if (dir_opt.size() && (mm_opt==1||mm_opt==2)) PrintErrorAndQuit("-mm 1 or 2 cannot be used with -dir");
        if (byresi_opt) PrintErrorAndQuit("-mm cannot be used with -byresi");
        if (ter_opt>=2 && (mm_opt==1 || mm_opt==2)) PrintErrorAndQuit("-mm 1 or 2 must be used with -ter 0 or -ter 1");
        if (mm_opt==4 && (yname.size() || dir2_opt.size()))
            cerr<<"WARNING! structure_2 is ignored for -mm 4"<<endl;
    }
    else if (full_opt) PrintErrorAndQuit("-full can only be used with -mm");

    if (o_opt && ter_opt<=1 && split_opt==2)
    {
        if (mm_opt && o_opt==2) cerr<<"WARNING! -mm may generate incorrect" 
            <<" RasMol output due to limitations in PDB file format. "
            <<"When -mm is used, -o is recommended over -rasmol"<<endl;
        else if (mm_opt==0) cerr<<"WARNING! Only the superposition of the"
            <<"last aligned chain pair will be generated"<<endl;
    }

    if (closeK_opt<0)
    {
        if (mm_opt==5) closeK_opt=5;
        else closeK_opt=0;
    }

    if (mm_opt==7 && hinge_opt>=10)
        PrintErrorAndQuit("ERROR! -hinge must be <10");


    /* read initial alignment file from 'align.txt' */
    if (i_opt) read_user_alignment(sequence, fname_lign, i_opt);

    if (byresi_opt==6) mm_opt=1;
    else if (byresi_opt) i_opt=3;

    if (m_opt && fname_matrix == "") // Output rotation matrix: matrix.txt
        PrintErrorAndQuit("ERROR! Please provide a file name for option -m!");

    /* parse file list */
    if (dir1_opt.size()+dir_opt.size()==0) chain1_list.push_back(xname);
    else file2chainlist(chain1_list, xname, dir_opt+dir1_opt, suffix_opt);

    int i; 
    if (dir_opt.size())
        for (i=0;i<chain1_list.size();i++)
            chain2_list.push_back(chain1_list[i]);
    else if (dir2_opt.size()==0) chain2_list.push_back(yname);
    else file2chainlist(chain2_list, yname, dir2_opt, suffix_opt);

    if (outfmt_opt==2)
    {
        if (mm_opt==2) cout<<"#Query\tTemplate\tTM"<<endl;
        else cout<<"#PDBchain1\tPDBchain2\tTM1\tTM2\t"
            <<"RMSD\tID1\tID2\tIDali\tL1\tL2\tLali"<<endl;
    }

    /* real alignment. entry functions are MMalign_main and 
     * TMalign_main */
    if (mm_opt==0) TMalign(xname, yname, fname_super, fname_lign, fname_matrix,
        sequence, Lnorm_ass, d0_scale, m_opt, i_opt, o_opt, a_opt,
        u_opt, d_opt, TMcut, infmt1_opt, infmt2_opt, ter_opt,
        split_opt, outfmt_opt, fast_opt, cp_opt, mirror_opt, het_opt,
        atom_opt, mol_opt, dir_opt, dir1_opt, dir2_opt, byresi_opt,
        chain1_list, chain2_list, se_opt);
    else if (mm_opt==1) MMalign(xname, yname, fname_super, fname_lign,
        fname_matrix, sequence, d0_scale, m_opt, o_opt,
        a_opt, d_opt, full_opt, TMcut, infmt1_opt, infmt2_opt,
        ter_opt, split_opt, outfmt_opt, fast_opt, mirror_opt, het_opt,
        atom_opt, mol_opt, dir1_opt, dir2_opt, chain1_list, chain2_list,
        byresi_opt);
    else if (mm_opt==2) MMdock(xname, yname, fname_super, 
        fname_matrix, sequence, Lnorm_ass, d0_scale, m_opt, o_opt, a_opt,
        u_opt, d_opt, TMcut, infmt1_opt, infmt2_opt, ter_opt,
        split_opt, outfmt_opt, fast_opt, mirror_opt, het_opt,
        atom_opt, mol_opt, dir1_opt, dir2_opt, chain1_list, chain2_list);
    else if (mm_opt==3) ; // should be changed to mm_opt=0, cp_opt=true
    else if (mm_opt==4) mTMalign(xname, yname, fname_super, fname_matrix,
        sequence, Lnorm_ass, d0_scale, m_opt, i_opt, o_opt, a_opt,
        u_opt, d_opt, full_opt, TMcut, infmt1_opt, ter_opt,
        split_opt, outfmt_opt, fast_opt, het_opt,
        atom_opt, mol_opt, dir_opt, byresi_opt, chain1_list);
    else if (mm_opt==5 || mm_opt==6) SOIalign(xname, yname, fname_super, fname_lign,
        fname_matrix, sequence, Lnorm_ass, d0_scale, m_opt, i_opt, o_opt,
        a_opt, u_opt, d_opt, TMcut, infmt1_opt, infmt2_opt, ter_opt,
        split_opt, outfmt_opt, fast_opt, cp_opt, mirror_opt, het_opt,
        atom_opt, mol_opt, dir_opt, dir1_opt, dir2_opt, 
        chain1_list, chain2_list, se_opt, closeK_opt, mm_opt);
    else cerr<<"WARNING! -mm "<<mm_opt<<" not implemented"<<endl;

    /* clean up */
    vector<string>().swap(chain1_list);
    vector<string>().swap(chain2_list);
    vector<string>().swap(sequence);

    t2 = clock();
    float diff = ((float)t2 - (float)t1)/CLOCKS_PER_SEC;
    if (outfmt_opt<2) printf("#Total CPU time is %5.2f seconds\n", diff);
    return 0;
}
