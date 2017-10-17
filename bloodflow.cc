#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_q_bubbles.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/base/timer.h>

#include <iostream>
#include <fstream>
#include <sstream>


namespace StokesBiot
  {
    using namespace dealii;

    using std::cout;
    using std::endl;

    template<int dim>
    struct PointHistory
      {
      // variables that appear with time derivative in the formulation
      Tensor<1, dim> old_stokes_velocity;
      Tensor<1, dim> old_displacement_face;
      double old_displacement_div;
      double old_darcy_pressure;
      };

    // Forward declaration
    template<int dim>
    class Postprocessor;

    template<int dim>
    class FluidStructureProblem
      {
    public:
      FluidStructureProblem(const unsigned int stokes_degree,
                            const unsigned int darcy_degree,
                            const unsigned int elasticity_degree,
                            const double time_step,
                            const unsigned int num_time_steps,
                            std::vector<Vector<double>> *sol = nullptr);

      void run(const unsigned int refine, std::string solname = "solution", bool compute_diff = 0);

      void set_vmodel(unsigned int val)
        { vmodel = val; }

      void reset();

    private:
      enum
        {
        fluid_domain_id,
        solid_domain_id
        };

      struct CellAssemblyScratchData
        {
        CellAssemblyScratchData (const hp::FECollection<dim> &fe,
                                 const hp::FECollection<dim> &fe_harmonic,
                                 const FiniteElement<dim> &stokes_fe,
                                 const FiniteElement<dim> &biot_fe,
                                 const hp::QCollection<dim> &q_collection,
                                 const hp::QCollection<dim> &q_collection_h,
                                 const Quadrature<dim-1>  &face_quadrature);
        CellAssemblyScratchData (const CellAssemblyScratchData &scratch_data);
        hp::FEValues<dim>     hp_fe_values;
        hp::FEValues<dim>     hp_fe_values_harmonic;

        FEFaceValues<dim> stokes_fe_face_values;
        FEFaceValues<dim> biot_fe_face_values;
        FESubfaceValues<dim> stokes_fe_subface_values;
        FESubfaceValues<dim> biot_fe_subface_values;
        };

      /*
       * Structure to copy data from threads to the main
       */
      struct CellAssemblyCopyData
        {
        CellAssemblyCopyData() : interface_flag(false) {}

        FullMatrix<double>                   cell_matrix;
        FullMatrix<double>  solid_interface, solid_bjs, fluid_interface, fs_bjs, ff_bjs;
        std::vector<types::global_dof_index> local_dof_indices, neighbor_fluid_dof_indices;

        bool interface_flag;
        };

      template<class CellType>
      static bool
      cell_is_in_fluid_domain(const CellType &cell);

      template<class CellType>
      static bool
      cell_is_in_solid_domain(const CellType &cell);

      void make_grid();

      void set_active_fe_indices();

      void setup_dofs();

      void assemble_system();
      void assemble_system_cell(const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
                                CellAssemblyScratchData                             &scratch,
                                CellAssemblyCopyData                                &copy_data);
      void copy_local_to_global (const CellAssemblyCopyData &copy_data);

      void assemble_rhs();

      void setup_quadrature_point_history();

      void update_quadrature_point_history();

      void assemble_fluid_interface_terms(const FEFaceValuesBase<dim> &biot_fe_face_values,
                                          const FEFaceValuesBase<dim> &stokes_fe_face_values,
                                          std::vector<double> &lm_phi,
                                          std::vector<Tensor<1, dim>> &stokes_phi_u,
                                          FullMatrix<double> &local_fluid_interface_matrix,
                                          FullMatrix<double> &local_fluid_fluid_BJS_matrix) const;

      void assemble_solid_interface_terms(const FEFaceValuesBase<dim> &biot_fe_face_values,
                                          const FEFaceValuesBase<dim> &stokes_fe_face_values,
                                          std::vector<double> &lm_phi,
                                          std::vector<Tensor<1, dim>> &darcy_phi_u,
                                          std::vector<Tensor<1, dim>> &disp_phi,
                                          FullMatrix<double> &local_solid_interface_matrix,
                                          FullMatrix<double> &local_solid_solid_BJS_matrix) const;

      void assemble_fluid_solid_BJS_term(const FEFaceValuesBase<dim> &biot_fe_face_values,
                                         const FEFaceValuesBase<dim> &stokes_fe_face_values,
                                         std::vector<Tensor<1, dim> > &stokes_phi_u,
                                         std::vector<Tensor<1, dim> > &disp_phi_u,
                                         FullMatrix<double> &local_fluid_solid_BJS_matrix) const;

      void assemble_old_fluid_solid_BJS_term(const int faceno,
                                             const FEFaceValuesBase<dim> &biot_fe_face_values,
                                             const FEFaceValuesBase<dim> &stokes_fe_face_values,
                                             std::vector<Tensor<1, dim> > &stokes_phi_u,
                                             const PointHistory<dim> *local_quadrature_points_data,
                                             Vector<double> &local_fluid_solid_BJS_vector) const;

      void assemble_old_solid_interface_terms(const int faceno,
                                              const FEFaceValuesBase<dim> &biot_fe_face_values,
                                              std::vector<Tensor<1, dim>> &disp_phi_u,
                                              std::vector<double> &lm_phi,
                                              const PointHistory<dim> *local_quadrature_points_data,
                                              Vector<double> &local_old_solid_interface_vector,
                                              Vector<double> &local_old_solid_solid_BJS_vector) const;

      void solve();

      void assemble_harmonic_extension_system();
      void assemble_harmonic_extension_rhs();
      void assemble_interface_rhs_terms_harmonic_extension(const hp::FEValues<dim> &true_fe_values,
                                                           const FEFaceValuesBase<dim> &harmonic_fe_face_values,
                                                           std::vector<Tensor<1, dim>> &mu_phi,
                                                           std::vector<Tensor<1, dim> > &disp_values,
                                                           Vector<double> &local_interface_vector) const;

      void solve_harmonic_extension();
      void move_mesh();
      void output_results(std::string solname = "solution", bool compute_diff = 0);

      const unsigned int stokes_degree;
      const unsigned int darcy_degree;
      const unsigned int elasticity_degree;

      // to construct fe with more than four spaces
      struct VectorElementDestroyer
        {
        VectorElementDestroyer(const std::vector<const FiniteElement<dim> *> &pointers);
        ~VectorElementDestroyer();

        const std::vector<const FiniteElement<dim> *> data;
        const std::vector<const FiniteElement<dim> *> &get_data() const;
        };

      static std::vector<const FiniteElement<dim> *> create_fe_list_fluid(const unsigned int stokes_degree);
      static std::vector<const FiniteElement<dim> *> create_fe_list_solid(const unsigned int darcy_degree,
                                                                          const unsigned int elasticity_degree);
      static std::vector<unsigned int> create_fe_multiplicities();

      Triangulation<dim> triangulation;
      FESystem<dim> stokes_fe;
      FESystem<dim> biot_fe;
      FESystem<dim> harmonic_extension_fe_fluid;
      FESystem<dim> harmonic_extension_fe_solid;
      hp::FECollection<dim> fe_collection;
      hp::FECollection<dim> fe_harmonic;
      hp::QCollection<dim> q_collection;
      hp::QCollection<dim> q_collection_harmonic;
      hp::DoFHandler<dim> dof_handler;
      hp::DoFHandler<dim> dof_handler_harmonic;

      SparsityPattern sparsity_pattern;
      SparsityPattern sparsity_pattern_harmonic;
      SparseMatrix<double> system_matrix;
      SparseMatrix<double> system_matrix_harmonic;

      std::vector<PointHistory<dim>> quadrature_point_history;

      Vector<double> solution;
      Vector<double> old_solution;
      Vector<double> harmonic_extension;
      Vector<double> old_harmonic_extension;
      Vector<double> previous_iteration_solution;
      Vector<double> incremental_displacement;
      Vector<double> system_rhs;
      Vector<double> harmonic_rhs;

      std::map<types::global_dof_index, double> boundary_values;
      std::map<types::global_dof_index, double> boundary_values_harmonic;

      //  const double viscosity;
      // cross model parameters in fluid region
      const double nu_f_0;
      const double nu_f_inf;
      const double K_f;
      const double r_f;
      // cross model parameters in solid region
      const double nu_p_0;
      const double nu_p_inf;
      const double K_p;
      const double r_p;
      // physical parameters
      const double alpha_bjs;
      const double lambda;
      const double mu;
      const double alpha_p;
      const double s_0;
      const double rho_f;
      const double a_f;
      const double a_p;
      // extra spring term in elasticity
      const double beta;

      double time;
      const double time_step;
      const unsigned int num_time_steps;

      //
      double viscosity_f(const Tensor<2, dim> &grad, unsigned int model = 0) const;
      double viscosity_s(const Tensor<1, dim> &vel, unsigned int model = 0) const;
      unsigned int vmodel;

      std::vector<Vector<double>> *solutions;
      friend Postprocessor<dim>;
      TimerOutput computing_timer;

      //////////////////////////////////////////////////////////////
      };


    // Not used in this test case!!!!
    ////////////////////////////////////////////////////////////////////////////////////
    template<int dim>
    class TrueSolution : public Function<dim>
      {
    private:
      const double current_time;
    public:
      TrueSolution(const double cur_time);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &value) const;

      virtual void vector_value_list(const std::vector<Point<dim>> &points,
                                     std::vector<Vector<double>> &value_list) const;

      inline double get_time()
        { return current_time; }
      };

    template<int dim>
    TrueSolution<dim>::TrueSolution(const double cur_time)
            :
            Function<dim>(3 * dim + 3),
            current_time(cur_time)
      {}

    template<int dim>
    void TrueSolution<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &values) const
      {
      double x = p[0], y = p[1];
      double pi = M_PI;
      switch (dim)
        {
        case 2:
          if (y >= 0)
            {
            // Stokes velocity
            values(0) = pi * cos(pi * current_time) * (-3 * x + cos(y));
            values(1) = pi * cos(pi * current_time) * (y + 1);
            // Stokes pressure
            values(2) = exp(current_time) * sin(pi * x) * cos(pi * y * 0.5) + 2 * pi * cos(pi * current_time);
            } else
            {
            // Stokes velocity
            values(0) = 0;
            values(1) = 0;
            // Stokes pressure
            values(2) = 0;
            }
          if (y <= 0)
            {
            // Darcy velocity
            values(3) = -pi * exp(current_time) * cos(pi * x) * cos(pi * y * 0.5);
            values(4) = pi * exp(current_time) * 0.5 * sin(pi * x) * sin(pi * y * 0.5);
            // Darcy pressure
            values(5) = exp(current_time) * sin(pi * x) * cos(pi * y * 0.5);
            // Displacement
            values(6) = sin(pi * current_time) * (-3 * x + cos(y));
            values(7) = sin(pi * current_time) * (y + 1);
            // LM (meaningless)
            values(8) = 0;
            } else
            {
            // Darcy velocity
            values(3) = 0;
            values(4) = 0;
            // Darcy pressure
            values(5) = 0;
            // Displacement
            values(6) = 0;
            values(7) = 0;
            // LM (meaningless)
            values(8) = 0;
            }

          break;
        case 3:
          values(0) = 0;
          values(1) = 0;
          values(2) = 0;

          // The rest is meaningless
          for (int i = 3; i < 2 * (dim + 1) + dim; ++i)
            values(i) = 0;

          break;
        default:
        Assert(false, ExcNotImplemented());
        }
      }

    template<int dim>
    void TrueSolution<dim>::vector_value_list(const std::vector<Point<dim>> &points,
                                              std::vector<Vector<double>> &value_list) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        TrueSolution<dim>::vector_value(points[p], value_list[p]);
      }
    ////////////////////////////////////////////////////////////////////////////////////

    template<int dim>
    class InitialCondition : public Function<dim>
      {
    public:
      InitialCondition() : Function<dim>(3 * dim + 3)
        {}

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &value) const;
      };


    template<int dim>
    void
    InitialCondition<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &values) const
      {

      for (unsigned int i = 0; i < 3 * dim + 3; ++i)
        values(i) = 0;

      }

    // In the Stokes region we set stress boundary conditions:
    // u_f \cdot \tau  is enforced essentially
    // p_in appears in the system naturally
    // as implemented now, won't work on general geometry
    template<int dim>
    class StokesVelocityBC : public Function<dim>
      {
    private:
      const double current_time;
    public:
      StokesVelocityBC(const double cur_time);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &value) const;

      virtual void vector_value_list(const std::vector<Point<dim>> &points,
                                     std::vector<Vector<double>> &value_list) const;

      inline double get_time()
        { return current_time; }
      };

    template<int dim>
    StokesVelocityBC<dim>::StokesVelocityBC(const double cur_time)
            :
            Function<dim>(3 * dim + 3),
            current_time(cur_time)
      {}

    template<int dim>
    void
    StokesVelocityBC<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &values) const
      {
      switch (dim)
        {
        case 2:
          // this is the tangetial component of u_f:
          values(1) = 0;

          // The rest is meaningless
          values(0) = 0;
          for (int i = 2; i < 3 * dim + 3; ++i)
            values(i) = 0;

          break;
        case 3:
          // here go tangential components:
          values(1) = 0;
          values(2) = 0;

          // The rest is meaningless
          values(0) = 0;
          for (int i = 3; i < 3 * dim + 3; ++i)
            values(i) = 0;

          break;
        default:
        Assert(false, ExcNotImplemented());
        }
      }

    template<int dim>
    void
    StokesVelocityBC<dim>::vector_value_list(const std::vector<Point<dim>> &points,
                                             std::vector<Vector<double>> &value_list) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        StokesVelocityBC<dim>::vector_value(points[p], value_list[p]);
      }


    template<int dim>
    class StokesPressureBC : public Function<dim>
      {
    private:
      const double current_time;
    public:
      StokesPressureBC(const double cur_time);

      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const;

      virtual void value_list(const std::vector<Point<dim>> &points,
                              std::vector<double> &value_list,
                              const unsigned int component = 0) const;

      };

    template<int dim>
    StokesPressureBC<dim>::StokesPressureBC(const double cur_time)
            :
            Function<dim>(1),
            current_time(cur_time)
      {}


    template<int dim>
    double StokesPressureBC<dim>::value(const Point<dim> &p,
                                        const unsigned int /*component*/) const
      {
      double pi = M_PI;
      switch (dim)
        {
        case 2:
          if (current_time <= 0.003)
            return 6667 * (1 - cos((2.0 * pi * current_time) / 0.003));
          else
            return 0;
          break;
        case 3:
          if (current_time <= 0.003)
            return 6667 * (1 - cos((2.0 * pi * current_time) / 0.003));
          else
            return 0;
          break;
        default:
        Assert (false, ExcNotImplemented());
        }
      }


    template<int dim>
    void StokesPressureBC<dim>::value_list(const std::vector<Point<dim>> &points,
                                           std::vector<double> &value_list,
                                           const unsigned int /*component = 0*/) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        value_list[p] = StokesPressureBC<dim>::value(points[p]);
      }


    template<int dim>
    class RightHandSideStokes : public Function<dim>
      {
    private:
      const double current_time;
    public:
      RightHandSideStokes(const double cur_time);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &value) const;

      virtual void vector_value_list(const std::vector<Point<dim>> &points,
                                     std::vector<Vector<double>> &value_list) const;

      };


    // the flow is driven by pressure BC,
    // so all right hand sides are zero
    template<int dim>
    RightHandSideStokes<dim>::RightHandSideStokes(const double cur_time)
            :
            Function<dim>(dim),
            current_time(cur_time)
      {}

    template<int dim>
    void
    RightHandSideStokes<dim>::vector_value(const Point<dim> &p,
                                           Vector<double> &value) const
      {
      switch (dim)
        {
        case 2:
          value(0) = 0;
          value(1) = 0;
          break;
        case 3:
          value(0) = 0;
          value(1) = 0;
          value(2) = 0;
          break;
        default:
        Assert (false, ExcNotImplemented());
        }
      }


    template<int dim>
    void
    RightHandSideStokes<dim>::vector_value_list(const std::vector<Point<dim>> &points,
                                                std::vector<Vector<double>> &value_list) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        RightHandSideStokes<dim>::vector_value(points[p], value_list[p]);
      }

    template<int dim>
    class SourceStokes : public Function<dim>
      {
    private:
      const double current_time;
    public:
      SourceStokes(const double cur_time);

      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const;

      virtual void value_list(const std::vector<Point<dim>> &points,
                              std::vector<double> &value_list,
                              const unsigned int component = 0) const;

      };

    template<int dim>
    SourceStokes<dim>::SourceStokes(const double cur_time)
            :
            Function<dim>(1),
            current_time(cur_time)
      {}


    template<int dim>
    double SourceStokes<dim>::value(const Point<dim> &p,
                                    const unsigned int /*component*/) const
      {
      switch (dim)
        {
        case 2:
          return 0;
          break;
        case 3:
          return 0;
          break;
        default:
        Assert (false, ExcNotImplemented());
        }
      }


    template<int dim>
    void SourceStokes<dim>::value_list(const std::vector<Point<dim>> &points,
                                       std::vector<double> &value_list,
                                       const unsigned int /*component = 0*/) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        value_list[p] = SourceStokes<dim>::value(points[p]);
      }


    // For Darcy system, we only impose velocity BC on left
    // and right parts of the solid domain
    // For velocity BC, we impose u_p \cdot n
    // As implemented right now, won't work in general
    template<int dim>
    class DarcyVelocityBC : public Function<dim>
      {
    private:
      const double current_time;
    public:
      DarcyVelocityBC(const double cur_time);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &value) const;

      virtual void vector_value_list(const std::vector<Point<dim>> &points,
                                     std::vector<Vector<double>> &value_list) const;

      inline double get_time()
        { return current_time; }
      };

    template<int dim>
    DarcyVelocityBC<dim>::DarcyVelocityBC(const double cur_time)
            :
            Function<dim>(3 * dim + 3),
            current_time(cur_time)
      {}

    template<int dim>
    void
    DarcyVelocityBC<dim>::vector_value(const Point<dim> &p,
                                       Vector<double> &values) const
      {
      switch (dim)
        {
        case 2:
          // this is the normal component of u_p:
          values(3) = 0;

          // The rest is meaningless
          for (int i = 0; i < 3 * dim + 3; ++i)
            if (i != 3)
              values(i) = 0;

          break;
        case 3:
          // this is the normal component of u_p:
          values(4) = 0;

          // The rest is meaningless
          for (int i = 0; i < 3 * dim + 3; ++i)
            if (i != 4)
              values(i) = 0;

          break;
        default:
        Assert(false, ExcNotImplemented());
        }
      }

    template<int dim>
    void
    DarcyVelocityBC<dim>::vector_value_list(const std::vector<Point<dim>> &points,
                                            std::vector<Vector<double>> &value_list) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        DarcyVelocityBC<dim>::vector_value(points[p], value_list[p]);
      }


    // DarcyPressureBC is not needed for this test case
    template<int dim>
    class DarcyPressureBC : public Function<dim>
      {
    private:
      const double current_time;
    public:
      DarcyPressureBC(const double cur_time);

      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const;

      virtual void value_list(const std::vector<Point<dim>> &points,
                              std::vector<double> &value_list,
                              const unsigned int component = 0) const;
      };

    template<int dim>
    DarcyPressureBC<dim>::DarcyPressureBC(const double cur_time)
            :
            Function<dim>(),
            current_time(cur_time)
      {}


    template<int dim>
    double DarcyPressureBC<dim>::value(const Point<dim> &p,
                                       const unsigned int /*component*/) const
      {
      switch (dim)
        {
        case 2:
          return 0;
          break;
        case 3:
          return 0;
          break;
        default:
        Assert(false, ExcNotImplemented());
        }
      }


    template<int dim>
    void
    DarcyPressureBC<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double> &value_list,
                                     const unsigned int /*component*/) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        value_list[p] = DarcyPressureBC<dim>::value(points[p]);
      }

    // strange BC:
    // on top and bottom we restrict only tangetial component
    // and on left and rigth - the whole vector
    template<int dim>
    class ElasticityBoundaryValues : public Function<dim>
      {
    private:
      const double current_time;
    public:
      ElasticityBoundaryValues(const double cur_time);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &value) const;

      virtual void vector_value_list(const std::vector<Point<dim>> &points,
                                     std::vector<Vector<double>> &value_list) const;
      };

    template<int dim>
    ElasticityBoundaryValues<dim>::ElasticityBoundaryValues(const double cur_time)
            :
            Function<dim>(3 * dim + 3),
            current_time(cur_time)
      {}


    template<int dim>
    void
    ElasticityBoundaryValues<dim>::vector_value(const Point<dim> &p,
                                                Vector<double> &values) const
      {
      switch (dim)
        {
        case 2:
          // when applying, choose proper values depending on the boundary part
          values(6) = 0;
          values(7) = 0;

          // The rest is meaningless
          for (int i = 0; i < 6; ++i)
            values(i) = 0;

          values(3 * dim + 2) = 0;

          break;
        case 3:
          values(8) = 0;
          values(9) = 0;
          values(10) = 0;

          // The rest is meaningless
          for (int i = 0; i < 8; ++i)
            values(i) = 0;
          values(3 * dim + 2) = 0;

          break;
        default:
        Assert(false, ExcNotImplemented());
        }
      }


    template<int dim>
    void
    ElasticityBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim>> &points,
                                                     std::vector<Vector<double>> &value_list) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        ElasticityBoundaryValues<dim>::vector_value(points[p], value_list[p]);
      }

    // in this test case the right hand sides are zero
    template<int dim>
    class RightHandSideBiot : public Function<dim>
      {
    private:
      const double current_time;
    public:
      RightHandSideBiot(const double cur_time);

      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const;

      virtual void value_list(const std::vector<Point<dim>> &points,
                              std::vector<double> &value_list,
                              const unsigned int component = 0) const;

      };

    template<int dim>
    RightHandSideBiot<dim>::RightHandSideBiot(const double cur_time)
            :
            Function<dim>(),
            current_time(cur_time)
      {}


    template<int dim>
    double
    RightHandSideBiot<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const
      {
      switch (dim)
        {
        case 2:
          return 0;
          break;
        case 3:
          return 0;
          break;
        default:
        Assert (false, ExcNotImplemented());
        }
      }


    template<int dim>
    void
    RightHandSideBiot<dim>::value_list(const std::vector<Point<dim>> &points,
                                       std::vector<double> &value_list,
                                       const unsigned int /*component = 0*/) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        value_list[p] = RightHandSideBiot<dim>::value(points[p]);
      }

    template<int dim>
    class RightHandSideDarcy : public Function<dim>
      {
    private:
      const double current_time;
    public:
      RightHandSideDarcy(const double cur_time);

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &value) const;

      virtual void vector_value_list(const std::vector<Point<dim>> &points,
                                     std::vector<Vector<double>> &value_list) const;

      };

    template<int dim>
    RightHandSideDarcy<dim>::RightHandSideDarcy(const double cur_time)
            :
            Function<dim>(dim),
            current_time(cur_time)
      {}

    template<int dim>
    void
    RightHandSideDarcy<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> &value) const
      {
      switch (dim)
        {
        case 2:
          value(0) = 0;
          value(1) = 0;
          break;
        case 3:
          value(0) = 0;
          value(1) = 0;
          value(2) = 0;
          break;
        default:
        Assert (false, ExcNotImplemented());
        }
      }


    template<int dim>
    void
    RightHandSideDarcy<dim>::vector_value_list(const std::vector<Point<dim>> &points,
                                               std::vector<Vector<double>> &value_list) const
      {
      for (unsigned int p = 0; p < points.size(); ++p)
        RightHandSideDarcy<dim>::vector_value(points[p], value_list[p]);
      }

    template<int dim>
    class KInverse : public TensorFunction<2, dim>
      {
    public:
      KInverse() : TensorFunction<2, dim>()
        {}

      virtual void value_list(const std::vector<Point<dim> > &points,
                              std::vector<Tensor<2, dim> > &values) const;
      };

    template<int dim>
    void
    KInverse<dim>::value_list(const std::vector<Point<dim> > &points,
                              std::vector<Tensor<2, dim> > &values) const
      {
      Assert (points.size() == values.size(),
              ExcDimensionMismatch(points.size(), values.size()));
      for (unsigned int p = 0; p < points.size(); ++p)
        {
        values[p].clear();
        for (unsigned int d = 0; d < dim; ++d)
          values[p][d][d] = 5e9 / 0.035;
        }
      }

    template<int dim>
    class Postprocessor : public DataPostprocessor<dim>
      {
    public:
      Postprocessor(const FluidStructureProblem<dim> *const ptr) : base(ptr)
        {}

      virtual
      void evaluate_vector_field
              (const DataPostprocessorInputs::Vector<dim> &inputs,
               std::vector<Vector<double>> &computed_quantities) const;

      virtual std::vector<std::string> get_names() const;

      virtual
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const;

      virtual
      UpdateFlags get_needed_update_flags() const;

    private:
      const FluidStructureProblem<dim> *const base;

      };

    template<int dim>
    std::vector<std::string>
    Postprocessor<dim>::get_names() const
      {
      std::vector<std::string> solution_names(dim, "stress1");
      for (unsigned int d = 1; d < dim; ++d)
        {
        std::string tmp = "stress";
        tmp += std::to_string(d + 1);

        for (unsigned dd = 0; dd < dim; ++dd)
          solution_names.push_back(tmp);
        }

      solution_names.push_back("viscosity_darcy");
      solution_names.push_back("viscosity_stokes");

//    // Viscosities only
//    std::vector<std::string> solution_names(1, "viscosity_darcy");
//    solution_names.push_back("viscosity_stokes");

      return solution_names;
      }

    template<int dim>
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    Postprocessor<dim>::get_data_component_interpretation() const
      {
//    // Viscosities only
//    std::vector<DataComponentInterpretation::DataComponentInterpretation>
//            interpretation (2, DataComponentInterpretation::component_is_scalar);


      std::vector<DataComponentInterpretation::DataComponentInterpretation>
              interpretation(dim * dim, DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);

      return interpretation;
      }

    template<int dim>
    UpdateFlags
    Postprocessor<dim>::get_needed_update_flags() const
      {
      return update_values | update_gradients | update_q_points;
      }

    template<int dim>
    void
    Postprocessor<dim>::evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &inputs,
                                              std::vector<Vector<double>> &computed_quantities) const
      {
      const unsigned int n_quadrature_points = inputs.solution_values.size();
      Assert (computed_quantities.size() == n_quadrature_points,
              ExcInternalError());

      for (unsigned int q = 0; q < n_quadrature_points; ++q)
        {
        Tensor<2, dim> grad_uf, grad_eta;
        Tensor<1, dim> up;
        for (unsigned int d = 0; d < dim; ++d)
          {
          grad_uf[d] = inputs.solution_gradients[q][d];
          up[d] = inputs.solution_values[q][dim + 1 + d];
          grad_eta[d] = inputs.solution_gradients[q][2 * dim + 2 + d];
          }

        const Tensor<2, dim> sym_grad_uf = symmetrize(grad_uf);
        const Tensor<2, dim> sym_grad_eta = symmetrize(grad_eta);

        // Identity matrix
        Tensor<2, dim> id;
        for (int i = 0; i < dim; ++i)
          id[i][i] = 1.0;

        const Tensor<2, dim> stress = 2.0 * (base->mu) * sym_grad_eta + (base->lambda) * trace(sym_grad_eta) * id;

        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            computed_quantities[q](i * dim + j) = stress[i][j];

        computed_quantities[q](dim * dim) = base->viscosity_s(up, base->vmodel);
        computed_quantities[q](dim * dim + 1) = base->viscosity_f(sym_grad_uf, base->vmodel);
        }
      }

    template<int dim>
    std::vector<const FiniteElement<dim> *>
    FluidStructureProblem<dim>::create_fe_list_fluid(const unsigned int stokes_degree)
      {
      std::vector<const FiniteElement<dim> *> fe_list;
      fe_list.push_back(new FE_Q<dim>(stokes_degree+1));
      fe_list.push_back(new FE_Q<dim>(stokes_degree));
      fe_list.push_back(new FE_Nothing<dim>(dim));
      fe_list.push_back(new FE_Nothing<dim>());
      fe_list.push_back(new FE_Nothing<dim>());
      fe_list.push_back(new FE_Nothing<dim>());
      return fe_list;
      }

    template<int dim>
    std::vector<const FiniteElement<dim> *>
    FluidStructureProblem<dim>::create_fe_list_solid(const unsigned int darcy_degree,
                                                     const unsigned int elasticity_degree)
      {
      std::vector<const FiniteElement<dim> *> fe_list;
      fe_list.push_back(new FE_Nothing<dim>());
      fe_list.push_back(new FE_Nothing<dim>());
      fe_list.push_back(new FE_RaviartThomasNodal<dim>(darcy_degree));
      fe_list.push_back(new FE_DGQ<dim>(darcy_degree));
      fe_list.push_back(new FE_Q<dim>(elasticity_degree));
      fe_list.push_back(new FE_DGQ<dim>(darcy_degree));
      return fe_list;
      }


    template<int dim>
    std::vector<unsigned int> FluidStructureProblem<dim>::create_fe_multiplicities()
      {
      std::vector<unsigned int> multiplicities;
      multiplicities.push_back(dim);
      multiplicities.push_back(1);
      multiplicities.push_back(1);
      multiplicities.push_back(1);
      multiplicities.push_back(dim);
      multiplicities.push_back(1);
      return multiplicities;
      }


    template<int dim>
    FluidStructureProblem<dim>::VectorElementDestroyer::VectorElementDestroyer(
            const std::vector<const FiniteElement<dim> *> &pointers)
            : data(pointers)
      {}

    template<int dim>
    FluidStructureProblem<dim>::VectorElementDestroyer::~VectorElementDestroyer()
      {
      for (unsigned int i = 0; i < data.size(); ++i)
        delete data[i];
      }

    template<int dim>
    const std::vector<const FiniteElement<dim> *> &FluidStructureProblem<dim>::VectorElementDestroyer::
    get_data() const
      {
      return data;
      }

    template<int dim>
    double
    FluidStructureProblem<dim>::viscosity_f(const Tensor<2, dim> &grad, unsigned int model) const
      {
      switch (model)
        {
        // Linear (Newtonian)
        case (0):
          return 0.035;
          // Carreau-Yasuda
        case (1):
          return nu_f_inf +
                 (nu_f_0 - nu_f_inf) * pow(1 + pow(K_f * sqrt(0.5 * scalar_product(grad, grad)), a_f), (r_f - 1) / a_f);
          // Cross model
        case (2):
          return (nu_f_inf + (nu_f_0 - nu_f_inf)
                             / (1 + K_f * K_f * 0.5 * std::pow(scalar_product(grad, grad), (1.0 - r_f) / 2.0)));
        default:
        Assert(false, ExcNotImplemented());
        }

      return 0;
      }

    template<int dim>
    double
    FluidStructureProblem<dim>::viscosity_s(const Tensor<1, dim> &vel, unsigned int model) const
      {
      switch (model)
        {
        // Linear (Newtonian)
        case (0):
          return 0.035;
          // Carreau-Yasuda
        case (1):
          return nu_p_inf + (nu_p_0 - nu_p_inf) * pow(1 + pow(K_p * sqrt(vel * vel), a_p), (r_p - 1) / a_p);
        case (2):
          return (nu_p_inf + (nu_p_0 - nu_p_inf)
                             / (1 + K_p * K_p * 0.5 * std::pow(vel * vel, (1.0 - r_p) / 2.0)));
        default:
        Assert(false, ExcNotImplemented());
        }

      return 0;
      }


    template<int dim>
    FluidStructureProblem<dim>::
    FluidStructureProblem(const unsigned int stokes_degree,
                          const unsigned int darcy_degree,
                          const unsigned int elasticity_degree,
                          const double time_step,
                          const unsigned int num_time_steps,
                          std::vector<Vector<double>> *sol)
            :
            stokes_degree(stokes_degree),
            darcy_degree(darcy_degree),
            elasticity_degree(elasticity_degree),
            triangulation(Triangulation<dim>::maximum_smoothing),
            stokes_fe(VectorElementDestroyer(create_fe_list_fluid(stokes_degree)).get_data(),
                      create_fe_multiplicities()),
            biot_fe(VectorElementDestroyer(create_fe_list_solid(darcy_degree, elasticity_degree)).get_data(),
                    create_fe_multiplicities()),
            harmonic_extension_fe_fluid(FE_Q<dim>(elasticity_degree), dim,
                                        FE_Q<dim>(elasticity_degree), dim),
            harmonic_extension_fe_solid(FE_Nothing<dim>(), dim,
                                        FE_Nothing<dim>(), dim),
            dof_handler(triangulation),
            dof_handler_harmonic(triangulation),
            // viscosity(0.035),
            nu_f_0(0.56),
            nu_f_inf(0.035),
            K_f(1.902),
            r_f(0.22),
            nu_p_0(0.56),
            nu_p_inf(0.035),
            K_p(1.902),
            r_p(0.22),
            alpha_bjs(sqrt(0.035 * 5e9)),
            lambda(4.28e6),
            mu(1.07e6),
            alpha_p(1.0),
            s_0(5e-6),
            rho_f(1.0),
            a_f(1.25),
            a_p(1.25),
            beta((3-dim)*5e7),    // extra spring term in elasticity
            time(0.0),
            time_step(time_step),
            num_time_steps(num_time_steps),
            vmodel(0),
            solutions(sol),
            computing_timer(std::cout, TimerOutput::summary,
                            TimerOutput::wall_times)
      {
      fe_collection.push_back(stokes_fe);
      fe_collection.push_back(biot_fe);

      fe_harmonic.push_back(harmonic_extension_fe_fluid);
      fe_harmonic.push_back(harmonic_extension_fe_solid);

      const QGauss<dim> stokes_quadrature(stokes_degree + 2);
      const QGauss<dim> biot_quadrature(std::max(darcy_degree, elasticity_degree) + 2);
      const QGauss<dim> harmonic_extension_quadrature(elasticity_degree + 2);

      q_collection.push_back(stokes_quadrature);
      q_collection.push_back(biot_quadrature);

      q_collection_harmonic.push_back(harmonic_extension_quadrature);
      q_collection_harmonic.push_back(harmonic_extension_quadrature);
      }


    template<int dim>
    template<class CellType>
    bool
    FluidStructureProblem<dim>::
    cell_is_in_fluid_domain(const CellType &cell)
      {
      return (cell->material_id() == fluid_domain_id);
      }

    template<int dim>
    template<class CellType>
    bool
    FluidStructureProblem<dim>::
    cell_is_in_solid_domain(const CellType &cell)
      {
      return (cell->material_id() == solid_domain_id);
      }

    template<int dim>
    Point<dim> cylinder_transform(const Point<dim> &p)
      {
      if (dim == 3)
        {
        Point<dim> q = p;
        if (!(p[1] == 0 && p[2] == 0))
          {
          q[1] = p[1] * std::max(std::abs(p[1]), std::abs(p[2])) / sqrt(p[1] * p[1] + p[2] * p[2]);
          q[2] = p[2] * std::max(std::abs(p[1]), std::abs(p[2])) / sqrt(p[1] * p[1] + p[2] * p[2]);
          }
        return q;
        } else if (dim == 2)
        return p;
      }

    // Function to transform the grid given a map
    template<int dim>
    Point<dim> grid_transform(const Point<dim> &p)
      {
      switch (dim)
        {
        case 2:
          return Point<dim>(p[0] + 0.03 * cos(M_PI * p[0]) * cos(M_PI * p[1]),
                            p[1] - 0.04 * cos(M_PI * p[0]) * cos(M_PI * p[1]));
          break;
        case 3:
          return Point<dim>(p[0] + 0.03 * cos(3 * M_PI * p[0]) * cos(3 * M_PI * p[1]) * cos(3 * M_PI * p[2]),
                            p[1] - 0.04 * cos(3 * M_PI * p[0]) * cos(3 * M_PI * p[1]) * cos(3 * M_PI * p[2]),
                            p[2] + 0.05 * cos(3 * M_PI * p[0]) * cos(3 * M_PI * p[1]) * cos(3 * M_PI * p[2]));
          break;
        default:
        Assert(false, ExcNotImplemented());
        }
      }

    template<int dim>
    void
    FluidStructureProblem<dim>::make_grid()
      {
      TimerOutput::Scope t(computing_timer, "Make grid");
      // Straight
      if (dim == 2)
        {
        Point<dim> p1_f, p2_f;
        p1_f[0] = 0.0;
        p1_f[1] = 0.1;
        p2_f[0] = 6.0;
        p2_f[1] = 1.1;

        const int nx = 48, ny = 8, mx = 48, my = 8;
        Triangulation<dim> tria_s_top;
        Triangulation<dim> tria_s_bot;
        Triangulation<dim> tria_f;

        std::vector<unsigned int> vec_f = {nx, ny};
        GridGenerator::subdivided_hyper_rectangle(tria_f, vec_f, p1_f, p2_f);


        Point<dim> p1_s_bot, p2_s_bot;
        p1_s_bot[0] = 0.0;
        p1_s_bot[1] = 0.0;
        p2_s_bot[0] = 6.0;
        p2_s_bot[1] = 0.1;
        std::vector<unsigned int> vec_s_bot = {mx, my};
        GridGenerator::subdivided_hyper_rectangle(tria_s_bot, vec_s_bot, p1_s_bot, p2_s_bot);

        Point<dim> p1_s_top, p2_s_top;
        p1_s_top[0] = 0.0;
        p1_s_top[1] = 1.1;
        p2_s_top[0] = 6.0;
        p2_s_top[1] = 1.2;
        std::vector<unsigned int> vec_s_top = {mx, my};
        GridGenerator::subdivided_hyper_rectangle(tria_s_top, vec_s_top, p1_s_top, p2_s_top);

        Triangulation<dim> tria_tmp;
        GridGenerator::merge_triangulations(tria_s_bot, tria_s_top, tria_tmp);
        GridGenerator::merge_triangulations(tria_tmp, tria_f, triangulation);

        for (typename Triangulation<dim>::active_cell_iterator cell = dof_handler.begin_active();
             cell != dof_handler.end(); ++cell)
          if (cell->center()[1] > 0.1 && cell->center()[1] < 1.1)
            cell->set_material_id(fluid_domain_id);
          else
            cell->set_material_id(solid_domain_id);

        for (typename Triangulation<dim>::active_cell_iterator cell = dof_handler_harmonic.begin_active();
             cell != dof_handler_harmonic.end(); ++cell)
          if (cell->center()[1] > 0.1 && cell->center()[1] < 1.1)
            cell->set_material_id(fluid_domain_id);
          else
            cell->set_material_id(solid_domain_id);

        for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
            // fluid domain:  [0,6] X [0.1,1.1]
            // solid domain:  [0,6] X ([0,0.1] U [1.1,1.2])
            // left(inflow) part of fluid domain
            if ((cell->face(f)->at_boundary())
                && (cell->face(f)->center()[1] > 0.1)
                && (cell->face(f)->center()[1] < 1.1)
                && (cell->face(f)->center()[0] < 1e-8))
              cell->face(f)->set_all_boundary_ids(1);
            // right(outflow) part of fluid domain
            if ((cell->face(f)->at_boundary())
                && (cell->face(f)->center()[1] > 0.1)
                && (cell->face(f)->center()[1] < 1.1)
                && (6.0 - cell->face(f)->center()[0] < 1e-8))
              cell->face(f)->set_all_boundary_ids(2);
            // left parts of solid domain
            if ((cell->face(f)->at_boundary())
                && (cell->face(f)->center()[1] < 0.1 || cell->face(f)->center()[1] > 1.1)
                && (cell->face(f)->center()[0] < 1e-8))
              cell->face(f)->set_all_boundary_ids(3);
            // right parts of solid domain
            if ((cell->face(f)->at_boundary())
                && (cell->face(f)->center()[1] < 0.1 || cell->face(f)->center()[1] > 1.1)
                && (6.0 - cell->face(f)->center()[0] < 1e-8))
              cell->face(f)->set_all_boundary_ids(3);
            // top and bottom parts of solid domain
            if ((cell->face(f)->at_boundary())
                && (1.2 - cell->face(f)->center()[1] < 1e-8 || cell->face(f)->center()[1] < 1e-8))
              cell->face(f)->set_all_boundary_ids(5);
            }
        }

      // Bifurcation
      if (dim == 2 && false)
        {
        Triangulation<dim> t1, t2, t3, t4, tria;

        GridIn<dim> gridin;
        gridin.attach_triangulation(t1);
        std::ifstream f("a.msh");
        gridin.read_msh(f);
        f.close();
        f.clear();

        gridin.attach_triangulation(t2);
        f.open("w1.msh");
        gridin.read_msh(f);
        f.close();
        f.clear();

        gridin.attach_triangulation(t3);
        f.open("w2.msh");
        gridin.read_msh(f);
        f.close();
        f.clear();

        gridin.attach_triangulation(t4);
        f.open("w3.msh");
        gridin.read_msh(f);
        f.close();
        f.clear();

        for (auto cell = t1.begin_active(); cell != t1.end(); ++cell)
          cell->set_material_id(fluid_domain_id);

        for (auto cell = t2.begin_active(); cell != t2.end(); ++cell)
          cell->set_material_id(solid_domain_id);

        for (auto cell = t3.begin_active(); cell != t3.end(); ++cell)
          cell->set_material_id(solid_domain_id);

        for (auto cell = t4.begin_active(); cell != t4.end(); ++cell)
          cell->set_material_id(solid_domain_id);


        GridGenerator::merge_triangulations(t1, t2, tria);
        GridGenerator::merge_triangulations(tria, t3, t1);
        GridGenerator::merge_triangulations(t1, t4, tria);

        triangulation.copy_triangulation(tria);

        for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
            // Set all boundary faces to 5 first, then change some
            if (cell->face(f)->at_boundary())
              cell->face(f)->set_all_boundary_ids(5);

            if ((cell->face(f)->at_boundary())
                && (cell->face(f)->center()[1] > -0.5)
                && (cell->face(f)->center()[1] < 0.5)
                && (cell->face(f)->center()[0] < 1e-8))
              cell->face(f)->set_all_boundary_ids(1);
            // right(outflow) part of fluid domain
            if ((cell->face(f)->at_boundary())
                && (((cell->face(f)->center()[1] > 1.0) && (cell->face(f)->center()[1] < 1.5))
                    || ((cell->face(f)->center()[1] > -1.5) && (cell->face(f)->center()[1] < -1.0)))
                && (6.0 - cell->face(f)->center()[0] < 1e-8))
              cell->face(f)->set_all_boundary_ids(2);
            // left parts of solid domain
            if ((cell->face(f)->at_boundary())
                && (cell->face(f)->center()[1] < -0.5 || cell->face(f)->center()[1] > 0.5)
                && (cell->face(f)->center()[0] < 1e-8))
              cell->face(f)->set_all_boundary_ids(3);
            // right parts of solid domain
            if ((cell->face(f)->at_boundary())
                && ((cell->face(f)->center()[1] < -1.5 || cell->face(f)->center()[1] > 1.5)
                    || (cell->face(f)->center()[1] < 1.0 || cell->face(f)->center()[1] > -1.0))
                && (6.0 - cell->face(f)->center()[0] < 1e-8))
              cell->face(f)->set_all_boundary_ids(3);
            }
        }

      if (dim == 3)
        {
          Triangulation<dim-1> t0, t1,t2,t3,t4,t5, tria;
          Triangulation<dim> tria3d;

          GridIn<dim-1> gridin;
          gridin.attach_triangulation(t0);
          std::ifstream f("q2.msh");
          gridin.read_msh(f);
          f.close();
          f.clear();

          t1.copy_triangulation(t0);

          GridTools::rotate(M_PI/2.0, t0);
          t2.copy_triangulation(t0);

          GridTools::rotate(M_PI/2.0, t0);
          t3.copy_triangulation(t0);

          GridTools::rotate(M_PI/2.0, t0);
          t4.copy_triangulation(t0);

          gridin.attach_triangulation(t5);
          f.open("out.msh");
          gridin.read_msh(f);
          f.close();
          f.clear();

          GridGenerator::merge_triangulations(t1, t2, t0);
          GridGenerator::merge_triangulations(t0, t3, t1);
          GridGenerator::merge_triangulations(t1, t4, t0);
          GridGenerator::merge_triangulations(t0, t5, tria);
          GridGenerator::extrude_triangulation(tria, 60, 6.0, tria3d);
          triangulation.copy_triangulation(tria3d);

          for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
            if (pow(cell->center()[0], 2) + pow(cell->center()[1], 2) > 0.25)
              cell->set_material_id(solid_domain_id);
            else
              cell->set_material_id(fluid_domain_id);

          for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
               cell != triangulation.end(); ++cell)
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
              // Set all faces to be 5, then change some
              if (cell->face(f)->at_boundary() 
                && cell_is_in_solid_domain(cell))
                cell->face(f)->set_all_boundary_ids(5);
              
              // bottom(inflow) part of fluid domain
              if ((cell->face(f)->at_boundary())
                  && (cell_is_in_fluid_domain(cell))
                  && (cell->face(f)->center()[2] < 1e-8))
                cell->face(f)->set_all_boundary_ids(1);

              // top(outflow) part of fluid domain
              if ((cell->face(f)->at_boundary())
                  && (cell_is_in_fluid_domain(cell))
                  && (cell->face(f)->center()[2] > 6.0 - 1e-8))
                cell->face(f)->set_all_boundary_ids(2);

              // bottom parts of solid domain
              if ((cell->face(f)->at_boundary())
                  && (cell_is_in_solid_domain(cell))
                  && (cell->face(f)->center()[2] < 1e-8))
                cell->face(f)->set_all_boundary_ids(3);

              // top parts of solid domain
              if ((cell->face(f)->at_boundary())
                  && (cell_is_in_solid_domain(cell))
                  && (cell->face(f)->center()[2] > 6.0 - 1e-8))
                cell->face(f)->set_all_boundary_ids(3);
              }
        }
      }


    template<int dim>
    void
    FluidStructureProblem<dim>::set_active_fe_indices()
      {
      TimerOutput::Scope t(computing_timer, "Set FE indices");

      for (typename hp::DoFHandler<dim>::active_cell_iterator
                   cell = dof_handler.begin_active();
           cell != dof_handler.end(); ++cell)
        {
        if (cell_is_in_fluid_domain(cell))
          cell->set_active_fe_index(0);
        else if (cell_is_in_solid_domain(cell))
          cell->set_active_fe_index(1);
        else
        Assert (false, ExcNotImplemented());
        }
      for (typename hp::DoFHandler<dim>::active_cell_iterator
                   cell = dof_handler_harmonic.begin_active();
           cell != dof_handler_harmonic.end(); ++cell)
        {
        if (cell_is_in_fluid_domain(cell))
          cell->set_active_fe_index(fluid_domain_id);
        else if (cell_is_in_solid_domain(cell))
          cell->set_active_fe_index(solid_domain_id);
        else
        Assert (false, ExcNotImplemented());
        }

      }

    template<int dim>
    void
    FluidStructureProblem<dim>::setup_dofs()
      {
      TimerOutput::Scope t(computing_timer, "Setup DOFs");

      set_active_fe_indices();
      dof_handler.distribute_dofs(fe_collection);
      dof_handler_harmonic.distribute_dofs(fe_harmonic);
      DoFRenumbering::component_wise(dof_handler);
      DoFRenumbering::component_wise(dof_handler_harmonic);

      std::cout << "   Number of active cells: "
                << triangulation.n_active_cells()
                << std::endl
                << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

      {
        DynamicSparsityPattern dsp(dof_handler.n_dofs(),
                                   dof_handler.n_dofs());

        Table<2, DoFTools::Coupling> cell_coupling(fe_collection.n_components(),
                                                   fe_collection.n_components());
        Table<2, DoFTools::Coupling> face_coupling(fe_collection.n_components(),
                                                   fe_collection.n_components());

        DynamicSparsityPattern dsp_h(dof_handler_harmonic.n_dofs(),
                                     dof_handler_harmonic.n_dofs());

        Table<2, DoFTools::Coupling> cell_coupling_h(fe_harmonic.n_components(),
                                                     fe_harmonic.n_components());
        Table<2, DoFTools::Coupling> face_coupling_h(fe_harmonic.n_components(),
                                                     fe_harmonic.n_components());

        for (unsigned int c = 0; c < fe_collection.n_components(); ++c)
          for (unsigned int d = 0; d < fe_collection.n_components(); ++d)
            {
            // Stokes system communication
            if ((c < dim + 1 && d < dim + 1) && !((c == dim) && (d == dim)))
              cell_coupling[c][d] = DoFTools::always;

            // Darcy system communication
            if ((c >= dim + 1 && c < 2 * dim + 2) && (d >= dim + 1 && d < 2 * dim + 2))
              cell_coupling[c][d] = DoFTools::always;

            // Displacement volume terms
            if ((c >= 2 * dim + 1 && c < 3 * dim + 2) && (d >= 2 * dim + 1 && d < 3 * dim + 2))
              cell_coupling[c][d] = DoFTools::always;

            // LM stab term
            cell_coupling[3 * dim + 2][3 * dim + 2] = DoFTools::always;

            // LM inteface B_gamma
            if ((c < dim || (c > dim && c < 2 * dim + 1) || (c > 2 * dim + 1 && c < 3 * dim + 2))
                && d == 3 * dim + 2)
              {
              face_coupling[c][d] = DoFTools::always;
              face_coupling[d][c] = DoFTools::always;
              }

            // BJS Stokes velocity - displacement coupling
            if (c < dim && (d > 2 * dim + 1 && d < 3 * dim + 2))
              {
              face_coupling[c][d] = DoFTools::always;
              face_coupling[d][c] = DoFTools::always;
              }
            }

        for (unsigned int c = 0; c < fe_harmonic.n_components(); ++c)
          for (unsigned int d = 0; d < fe_harmonic.n_components(); ++d)
            {
            if (c < dim && d < dim)
              cell_coupling_h[c][d] = DoFTools::always;

            if (c >= dim && d >= dim)
              cell_coupling_h[c][d] = DoFTools::always;

            if (c < dim && d >= dim)
              {
              face_coupling_h[c][d] = DoFTools::always;
              face_coupling_h[d][c] = DoFTools::always;
              }

            }

        DoFTools::make_flux_sparsity_pattern(dof_handler, dsp,
                                             cell_coupling, face_coupling);

        DoFTools::make_flux_sparsity_pattern(dof_handler_harmonic, dsp_h,
                                             cell_coupling_h, face_coupling_h);
        sparsity_pattern.copy_from(dsp);
        sparsity_pattern_harmonic.copy_from(dsp_h);
      }

      setup_quadrature_point_history();

      system_matrix.reinit(sparsity_pattern);
      system_matrix_harmonic.reinit(sparsity_pattern_harmonic);

      solution.reinit(dof_handler.n_dofs());
      harmonic_extension.reinit(dof_handler_harmonic.n_dofs());
      old_harmonic_extension.reinit(dof_handler_harmonic.n_dofs());
      system_rhs.reinit(dof_handler.n_dofs());
      harmonic_rhs.reinit(dof_handler_harmonic.n_dofs());
      old_solution.reinit(dof_handler.n_dofs());
      incremental_displacement.reinit(dof_handler.n_dofs());
      previous_iteration_solution.reinit(dof_handler.n_dofs());

      const FEValuesExtractors::Scalar stokes_velocities_tangential_x(0);
      const FEValuesExtractors::Scalar stokes_velocities_tangential_y(1);
      const FEValuesExtractors::Vector displacement(2 * dim + 2);

      cout << " Projecting Darcy BC ..." << endl;
      typename FunctionMap<dim>::type darcy_velocity_bc;
      DarcyVelocityBC<dim> darcy_bc_func(time);

      darcy_velocity_bc[3] = &darcy_bc_func;

      const QGauss<dim-1> stokes_face_quadrature(stokes_degree + 2);
      const QGauss<dim-1> biot_face_quadrature(std::max(darcy_degree, elasticity_degree) + 2);

      hp::QCollection<dim - 1> q_face_collection;
      q_face_collection.push_back(stokes_face_quadrature);
      q_face_collection.push_back(biot_face_quadrature);

      VectorTools::project_boundary_values(dof_handler,
                                           darcy_velocity_bc,
                                           q_face_collection,
                                           boundary_values);

      typename FunctionMap<dim>::type stokes_velocity_bc;
      StokesVelocityBC<dim> stokes_bc_func(time);
      stokes_velocity_bc[1] = &stokes_bc_func;
      stokes_velocity_bc[2] = &stokes_bc_func;

      cout << " Preinterpolating Stokes, Elasticity and Harmonic BC ..." << endl;
      VectorTools::interpolate_boundary_values(dof_handler,
                                               stokes_velocity_bc,
                                               boundary_values,
                                               fe_collection.component_mask(stokes_velocities_tangential_x));
                                               
      VectorTools::interpolate_boundary_values(dof_handler,
                                               stokes_velocity_bc,
                                               boundary_values,
                                               fe_collection.component_mask(stokes_velocities_tangential_y));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               ElasticityBoundaryValues<dim>(time),
                                               boundary_values,
                                               fe_collection.component_mask(displacement));


      typename FunctionMap<dim>::type harmonic_extension_bc;
      ZeroFunction<dim> zero_func(2*dim);
      harmonic_extension_bc[1] = &zero_func;
      harmonic_extension_bc[2] = &zero_func;
      const FEValuesExtractors::Vector h_ext(0);

      VectorTools::interpolate_boundary_values(dof_handler_harmonic,
                                               harmonic_extension_bc,
                                               boundary_values_harmonic,
                                               fe_harmonic.component_mask(h_ext));
      }


    template <int dim>
    FluidStructureProblem<dim>::CellAssemblyScratchData::
    CellAssemblyScratchData (const hp::FECollection<dim> &fe,
                             const hp::FECollection<dim> &fe_harmonic,
                             const FiniteElement<dim> &stokes_fe,
                             const FiniteElement<dim> &biot_fe,
                             const hp::QCollection<dim> &q_collection,
                             const hp::QCollection<dim> &q_collection_h,
                             const Quadrature<dim-1>  &face_quadrature)
            :
            hp_fe_values(fe, q_collection,
                         update_values | update_quadrature_points |
                         update_JxW_values | update_gradients),
            hp_fe_values_harmonic(fe_harmonic, q_collection_h,
                                  update_values),
            stokes_fe_face_values(stokes_fe, face_quadrature,
                                  update_values | update_JxW_values |
                                  update_normal_vectors | update_gradients),
            biot_fe_face_values(biot_fe, face_quadrature,
                                update_values | update_JxW_values |
                                update_quadrature_points | update_normal_vectors |
                                update_gradients),
            stokes_fe_subface_values(stokes_fe, face_quadrature,
                                     update_JxW_values | update_normal_vectors |
                                     update_gradients),
            biot_fe_subface_values(biot_fe, face_quadrature, update_values)
      {}


    template <int dim>
    FluidStructureProblem<dim>::CellAssemblyScratchData::
    CellAssemblyScratchData (const CellAssemblyScratchData &scratch_data)
            :
            hp_fe_values (scratch_data.hp_fe_values.get_fe_collection(),
                          scratch_data.hp_fe_values.get_quadrature_collection(),
                          update_values   | update_gradients |
                          update_quadrature_points | update_JxW_values),
            hp_fe_values_harmonic(scratch_data.hp_fe_values_harmonic.get_fe_collection(),
                                  scratch_data.hp_fe_values_harmonic.get_quadrature_collection(),
                                  update_values),
            stokes_fe_face_values(scratch_data.stokes_fe_face_values.get_fe(),
                                  scratch_data.stokes_fe_face_values.get_quadrature(),
                                  update_values | update_JxW_values |
                                  update_normal_vectors | update_gradients),
            biot_fe_face_values(scratch_data.biot_fe_face_values.get_fe(),
                                scratch_data.biot_fe_face_values.get_quadrature(),
                                update_values | update_JxW_values |
                                update_quadrature_points | update_normal_vectors |
                                update_gradients),
            stokes_fe_subface_values(scratch_data.stokes_fe_subface_values.get_fe(),
                                     scratch_data.stokes_fe_subface_values.get_quadrature(),
                                     update_JxW_values | update_normal_vectors |
                                     update_gradients),
            biot_fe_subface_values(scratch_data.biot_fe_subface_values.get_fe(),
                                   scratch_data.biot_fe_subface_values.get_quadrature(),
                                   update_values)
      {}

    template <int dim>
    void FluidStructureProblem<dim>::assemble_system()
      {
      TimerOutput::Scope t(computing_timer, "Assemble system");
      const QGauss<dim-1> common_face_quadrature(std::max(std::max(stokes_degree + 2,
                                                                   darcy_degree + 2), elasticity_degree + 2));

      WorkStream::run(dof_handler.begin_active(),
                      dof_handler.end(),
                      *this,
                      &FluidStructureProblem::assemble_system_cell,
                      &FluidStructureProblem::copy_local_to_global,
                      CellAssemblyScratchData(fe_collection,
                                              fe_harmonic,
                                              stokes_fe,
                                              biot_fe,
                                              q_collection,
                                              q_collection_harmonic,
                                              common_face_quadrature),
                      CellAssemblyCopyData());

      }

    template <int dim>
    void FluidStructureProblem<dim>::copy_local_to_global (const CellAssemblyCopyData &copy_data)
      {
      // put volume terms into global matrix
      for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
        for (unsigned int j = 0; j < copy_data.local_dof_indices.size(); ++j)
          system_matrix.add(copy_data.local_dof_indices[i], copy_data.local_dof_indices[j], copy_data.cell_matrix(i, j));


      if (copy_data.neighbor_fluid_dof_indices.size() != 0)
        {

        //std::cout << "What is going on: " << copy_data.local_dof_indices.size() << endl;

        for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
          {
          for (unsigned int j = 0; j < copy_data.local_dof_indices.size(); ++j)
            {
            system_matrix.add(copy_data.local_dof_indices[i],
                              copy_data.local_dof_indices[j],
                              copy_data.solid_interface(i, j));
            system_matrix.add(copy_data.local_dof_indices[i],
                              copy_data.local_dof_indices[j],
                              copy_data.solid_bjs(i, j));
            }

          for (unsigned int j = 0; j < copy_data.neighbor_fluid_dof_indices.size(); ++j)
            {
            system_matrix.add(copy_data.local_dof_indices[i],
                              copy_data.neighbor_fluid_dof_indices[j],
                              copy_data.fluid_interface(j, i));
            system_matrix.add(copy_data.neighbor_fluid_dof_indices[j],
                              copy_data.local_dof_indices[i],
                              copy_data.fluid_interface(j, i));
            // term <u_f, \xi_p> :
            system_matrix.add(copy_data.local_dof_indices[i],
                              copy_data.neighbor_fluid_dof_indices[j],
                              time_step * copy_data.fs_bjs(j, i));
            // term <v_f, \d_t \eta_p> :
            system_matrix.add(copy_data.neighbor_fluid_dof_indices[j],
                              copy_data.local_dof_indices[i],
                              copy_data.fs_bjs(j, i));
            }
          }

        for (unsigned int i = 0; i < copy_data.neighbor_fluid_dof_indices.size(); ++i)
          for (unsigned int j = 0; j < copy_data.neighbor_fluid_dof_indices.size(); ++j)
            {

            system_matrix.add(copy_data.neighbor_fluid_dof_indices[i],
                              copy_data.neighbor_fluid_dof_indices[j],
                              copy_data.ff_bjs(i, j));
            }
        }
      }

    template<int dim>
    void FluidStructureProblem<dim>::assemble_system_cell(const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
                                                          CellAssemblyScratchData                              &scratch_data,
                                                          CellAssemblyCopyData                                 &copy_data)
      {
      //TimerOutput::Scope t(computing_timer, "Assemble system, cell");
      //system_matrix = 0;

      const unsigned int stokes_dofs_per_cell = stokes_fe.dofs_per_cell;
      const unsigned int biot_dofs_per_cell = biot_fe.dofs_per_cell;

      copy_data.solid_interface.reinit(biot_dofs_per_cell, biot_dofs_per_cell);
      copy_data.solid_bjs.reinit(biot_dofs_per_cell, biot_dofs_per_cell);
      copy_data.fluid_interface.reinit(stokes_dofs_per_cell, stokes_dofs_per_cell);
      copy_data.fs_bjs.reinit(stokes_dofs_per_cell, biot_dofs_per_cell);
      copy_data.ff_bjs.reinit(stokes_dofs_per_cell, stokes_dofs_per_cell);

      //std::vector<types::global_dof_index> local_dof_indices;

      const KInverse<dim> k_inverse;

      const FEValuesExtractors::Vector stokes_velocities(0);
      const FEValuesExtractors::Scalar stokes_pressure(dim);
      const FEValuesExtractors::Vector darcy_velocities(dim + 1);
      const FEValuesExtractors::Scalar darcy_pressure(2 * dim + 1);
      const FEValuesExtractors::Vector displacement(2 * dim + 2);
      const FEValuesExtractors::Scalar l_multiplier(3 * dim + 2);

      // Stokes region test functions
      std::vector<Tensor<2, dim> > stokes_sym_grad_phi_u(stokes_dofs_per_cell);
      std::vector<Tensor<2, dim> > stokes_grad_phi_u(stokes_dofs_per_cell);
      std::vector<Tensor<1, dim> > stokes_phi_u(stokes_dofs_per_cell);
      std::vector<double> stokes_div_phi_u(stokes_dofs_per_cell);
      std::vector<double> stokes_phi_p(stokes_dofs_per_cell);
      // Biot region test functions (Darcy)
      std::vector<Tensor<1, dim>> darcy_phi_u(biot_dofs_per_cell);
      std::vector<double> darcy_div_phi_u(biot_dofs_per_cell);
      std::vector<double> darcy_phi_p(biot_dofs_per_cell);
      std::vector<Tensor<1, dim>> darcy_grad_phi_p(biot_dofs_per_cell);
      // Biot region test functions (Elasticity)
      std::vector<Tensor<2, dim>> disp_sym_grad_phi(biot_dofs_per_cell);
      std::vector<Tensor<1, dim>> disp_phi_u(biot_dofs_per_cell);
      std::vector<double> disp_div_phi(biot_dofs_per_cell);
      // Biot region test functions (Lagrange multiplier)
      std::vector<double> lm_phi(biot_dofs_per_cell);

      scratch_data.hp_fe_values.reinit(cell);
      const FEValues<dim> &fe_values = scratch_data.hp_fe_values.get_present_fe_values();

      copy_data.cell_matrix.reinit(cell->get_fe().dofs_per_cell, cell->get_fe().dofs_per_cell);
      copy_data.local_dof_indices.resize(cell->get_fe().dofs_per_cell);

      cell->get_dof_indices(copy_data.local_dof_indices);

      if (cell_is_in_fluid_domain(cell))
        {
        copy_data.neighbor_fluid_dof_indices.resize(0);
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        Assert (dofs_per_cell == stokes_dofs_per_cell,
                ExcInternalError());

        std::vector<Tensor<2, dim>> prev_iteration_stokes_sym_grad(fe_values.n_quadrature_points);
        fe_values[stokes_velocities].get_function_gradients(previous_iteration_solution,
                                                            prev_iteration_stokes_sym_grad);

        std::vector<Tensor<1, dim>> prev_iteration_stokes_values(fe_values.n_quadrature_points);
        fe_values[stokes_velocities].get_function_values(previous_iteration_solution, prev_iteration_stokes_values);

        typename hp::DoFHandler<dim>::active_cell_iterator data_cell(&triangulation, cell->level(),
                                                                     cell->index(),
                                                                     &dof_handler_harmonic);
        scratch_data.hp_fe_values_harmonic.reinit(data_cell);
        const FEValuesExtractors::Vector h_ext(0);
        const FEValues<dim> &h_fe_values = scratch_data.hp_fe_values_harmonic.get_present_fe_values();

        std::vector<Tensor<1, dim>> h_values(fe_values.n_quadrature_points);
        h_fe_values[h_ext].get_function_values(harmonic_extension, h_values);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          {
          prev_iteration_stokes_sym_grad[q] =
                  0.5 * (prev_iteration_stokes_sym_grad[q] + transpose(prev_iteration_stokes_sym_grad[q]));
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
            stokes_sym_grad_phi_u[k] = fe_values[stokes_velocities].symmetric_gradient(k, q);
            stokes_grad_phi_u[k] = fe_values[stokes_velocities].gradient(k, q);
            stokes_phi_u[k] = fe_values[stokes_velocities].value(k, q);
            stokes_div_phi_u[k] = fe_values[stokes_velocities].divergence(k, q);
            stokes_phi_p[k] = fe_values[stokes_pressure].value(k, q);
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              copy_data.cell_matrix(i, j) +=
                      ((rho_f / time_step) * stokes_phi_u[i] * stokes_phi_u[j]
                       + stokes_grad_phi_u[j] * (prev_iteration_stokes_values[q] - h_values[q]) * stokes_phi_u[i]
                       + 2.0 * viscosity_f(prev_iteration_stokes_sym_grad[q], vmodel)
                         * scalar_product(stokes_sym_grad_phi_u[i], stokes_sym_grad_phi_u[j])
                       - stokes_div_phi_u[i] * stokes_phi_p[j]
                       + stokes_phi_p[i] * stokes_div_phi_u[j])
                      * fe_values.JxW(q);
            }
          }
        }
      else
        {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        std::vector<Tensor<2, dim> > k_inverse_values(fe_values.n_quadrature_points);
        Assert (dofs_per_cell == biot_dofs_per_cell,
                ExcInternalError());

        k_inverse.value_list(fe_values.get_quadrature_points(),
                             k_inverse_values);

        std::vector<Tensor<1, dim>> prev_iteration_darcy_values(fe_values.n_quadrature_points);
        fe_values[darcy_velocities].get_function_values(previous_iteration_solution, prev_iteration_darcy_values);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          {
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
            // Darcy velocity and pressures
            darcy_phi_u[k] = fe_values[darcy_velocities].value(k, q);
            darcy_div_phi_u[k] = fe_values[darcy_velocities].divergence(k, q);
            darcy_phi_p[k] = fe_values[darcy_pressure].value(k, q);
            lm_phi[k] = fe_values[l_multiplier].value(k, q);

            // Displacements
            disp_sym_grad_phi[k] = fe_values[displacement].symmetric_gradient(k, q);
            disp_phi_u[k] = fe_values[displacement].value(k, q);
            disp_div_phi[k] = fe_values[displacement].divergence(k, q);
            }


          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              copy_data.cell_matrix(i, j) += ((s_0 / time_step) * darcy_phi_p[i] * darcy_phi_p[j]
                                              + viscosity_s(prev_iteration_darcy_values[q], vmodel) * darcy_phi_u[i] *
                                                k_inverse_values[q] * darcy_phi_u[j]
                                              - darcy_div_phi_u[i] * darcy_phi_p[j]
                                              + darcy_phi_p[i] * darcy_div_phi_u[j]
                                              + 2.0 * mu * scalar_product(disp_sym_grad_phi[i], disp_sym_grad_phi[j])
                                              + lambda * disp_div_phi[i] * disp_div_phi[j]
                                              - disp_div_phi[i] * darcy_phi_p[j]
                                              + (1.0 / time_step) * disp_div_phi[j] * darcy_phi_p[i]
                                              + 1.e-8 * lm_phi[i] * lm_phi[j]
                                              + beta * disp_phi_u[i] * disp_phi_u[j]) * fe_values.JxW(q);
            }
          }
        }

      // assemble interface terms
      if (cell_is_in_solid_domain(cell))
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->at_boundary(f) == false)
            {
            copy_data.neighbor_fluid_dof_indices.resize(stokes_dofs_per_cell);


            // if neighbor is in fluid domain and is on the same level of refinement
            // and doesn't have children
            if ((cell->neighbor(f)->level() == cell->level())
                && (cell->neighbor(f)->has_children() == false)
                && (cell_is_in_fluid_domain(cell->neighbor(f))))
              {
              scratch_data.biot_fe_face_values.reinit(cell, f);
              scratch_data.stokes_fe_face_values.reinit(cell->neighbor(f), cell->neighbor_of_neighbor(f));

              // neighbor_fluid_dof_indices.resize (cell->neighbor(f)->get_fe().dofs_per_cell);
              cell->neighbor(f)->get_dof_indices(copy_data.neighbor_fluid_dof_indices);

              // b-gamma and bjs coming from solid-solid interaction:
              assemble_solid_interface_terms(scratch_data.biot_fe_face_values,
                                             scratch_data.stokes_fe_face_values,
                                             lm_phi, darcy_phi_u, disp_phi_u,
                                             copy_data.solid_interface,
                                             copy_data.solid_bjs);


              // b-gamma and bjs coming from fluid-solid interaction:
              assemble_fluid_interface_terms(scratch_data.biot_fe_face_values,
                                             scratch_data.stokes_fe_face_values,
                                             lm_phi, stokes_phi_u,
                                             copy_data.fluid_interface,
                                             copy_data.ff_bjs);

              assemble_fluid_solid_BJS_term(scratch_data.biot_fe_face_values,
                                            scratch_data.stokes_fe_face_values,
                                            stokes_phi_u, disp_phi_u,
                                            copy_data.fs_bjs);

              copy_data.interface_flag = true;
              }
            }
      }


    template<int dim>
    void FluidStructureProblem<dim>::assemble_harmonic_extension_system()
      {
      TimerOutput::Scope t(computing_timer, "Assemble Harmonic Extension System");
      system_matrix_harmonic = 0;

      hp::FEValues<dim> hp_fe_values(fe_harmonic, q_collection_harmonic,
                                     update_values |
                                     update_quadrature_points |
                                     update_JxW_values |
                                     update_gradients);

      const QGauss<dim - 1> face_quadrature(elasticity_degree + 2);

      FEFaceValues<dim> fe_face_values(harmonic_extension_fe_fluid,
                                       face_quadrature,
                                       update_values |
                                       update_JxW_values |
                                       update_normal_vectors |
                                       update_gradients);


      FESubfaceValues<dim> fe_subface_values(harmonic_extension_fe_fluid,
                                             face_quadrature,
                                             update_JxW_values |
                                             update_normal_vectors |
                                             update_gradients);

      const unsigned int dofs_per_cell_h = harmonic_extension_fe_fluid.dofs_per_cell;

      FullMatrix<double> local_matrix;
      //FullMatrix<double> local_interface_matrix(harmonic_extension_fe_fluid.dofs_per_cell,
      //                                          harmonic_extension_fe_fluid.dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices;

      const FEValuesExtractors::Vector disp(0);
      const FEValuesExtractors::Vector lm(dim);

      std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell_h);
      std::vector<Tensor<1, dim>> h_phi_u(dofs_per_cell_h);
      std::vector<Tensor<1, dim>> mu_phi(dofs_per_cell_h);


      typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler_harmonic.begin_active(),
              endc = dof_handler_harmonic.end();
      for (; cell != endc; ++cell)
        {
        hp_fe_values.reinit(cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        local_matrix.reinit(cell->get_fe().dofs_per_cell, cell->get_fe().dofs_per_cell);

        local_dof_indices.resize(cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        // assemble volume terms
        if (cell_is_in_fluid_domain(cell))
          {
          const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
          Assert (dofs_per_cell == dofs_per_cell_h,
                  ExcInternalError());

          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
              grad_phi_u[k] = fe_values[disp].gradient(k, q);
              mu_phi[k] = fe_values[lm].value(k, q);
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                local_matrix(i, j) += (scalar_product(grad_phi_u[i], grad_phi_u[j])
                                       + 1e-8 * mu_phi[i] * mu_phi[j]) * fe_values.JxW(q);
              }
            }
          }

        // put volume terms into global matrix
        for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
          for (unsigned int j = 0; j < cell->get_fe().dofs_per_cell; ++j)
            system_matrix_harmonic.add(local_dof_indices[i], local_dof_indices[j], local_matrix(i, j));

        // assemble interface terms
        if (cell_is_in_fluid_domain(cell))
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) == false && cell_is_in_solid_domain(cell->neighbor(f)))
              {

              fe_face_values.reinit(cell, f);
              local_matrix.reinit(fe_face_values.dofs_per_cell, fe_face_values.dofs_per_cell);
              local_matrix = 0;

              for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
                {
                for (unsigned int k = 0; k < fe_face_values.dofs_per_cell; ++k)
                  {
                  h_phi_u[k] = fe_face_values[disp].value(k, q);
                  mu_phi[k] = fe_face_values[lm].value(k, q);
                  }

                for (unsigned int i = 0; i < fe_face_values.dofs_per_cell; ++i)
                  {
                  for (unsigned int j = 0; j < fe_face_values.dofs_per_cell; ++j)
                    {
                    local_matrix(i, j) +=
                            (h_phi_u[i] * mu_phi[j] + h_phi_u[j] * mu_phi[i]) * fe_face_values.JxW(q);
                    }
                  }

                }

              for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                for (unsigned int j = 0; j < cell->get_fe().dofs_per_cell; ++j)
                  {
                  system_matrix_harmonic.add(local_dof_indices[i], local_dof_indices[j], local_matrix(i, j));
                  }

              }
        }

      //    std::ofstream h_out("matrixHarmonic.txt");
      //    system_matrix_harmonic.print_formatted(h_out, 3, 1, 1, "0");
      //    h_out.close();
      }


    template<int dim>
    void FluidStructureProblem<dim>::assemble_rhs()
      {
      TimerOutput::Scope t(computing_timer, "Assemble RHS");
      system_rhs = 0;

      hp::FEValues<dim> hp_fe_values(fe_collection, q_collection,
                                     update_values |
                                     update_quadrature_points |
                                     update_JxW_values |
                                     update_gradients);

      const QGauss<dim - 1> common_face_quadrature(std::max(std::max(stokes_degree + 2, darcy_degree + 2),
                                                            elasticity_degree + 2));

      FEFaceValues<dim> stokes_fe_face_values(stokes_fe,
                                              common_face_quadrature,
                                              update_values |
                                              update_JxW_values |
                                              update_quadrature_points |
                                              update_normal_vectors |
                                              update_gradients);
      FEFaceValues<dim> biot_fe_face_values(biot_fe,
                                            common_face_quadrature,
                                            update_values |
                                            update_JxW_values |
                                            update_quadrature_points |
                                            update_normal_vectors |
                                            update_gradients);

      FESubfaceValues<dim> stokes_fe_subface_values(stokes_fe,
                                                    common_face_quadrature,
                                                    update_JxW_values |
                                                    update_normal_vectors |
                                                    update_gradients);
      FESubfaceValues<dim> biot_fe_subface_values(biot_fe,
                                                  common_face_quadrature,
                                                  update_values);


      const unsigned int stokes_dofs_per_cell = stokes_fe.dofs_per_cell;
      const unsigned int biot_dofs_per_cell = biot_fe.dofs_per_cell;


      Vector<double> local_rhs;

      std::vector<types::global_dof_index> local_dof_indices;

      //cout << "Time in Assemble rhs: " << time << endl;
      // const RightHandSideStokes<dim> right_hand_side_stokes(time);
      // const SourceStokes<dim> source_stokes(time);

      // const RightHandSideBiot<dim> right_hand_side_biot(time);
      // const RightHandSideDarcy<dim> right_hand_side_darcy(time);

      const DarcyPressureBC<dim> darcy_pressure_bc(time);
      const StokesPressureBC<dim> stokes_pressure_bc(time);

      const FEValuesExtractors::Vector stokes_velocities(0);
      const FEValuesExtractors::Scalar stokes_pressure(dim);
      const FEValuesExtractors::Vector darcy_velocities(dim + 1);
      const FEValuesExtractors::Scalar darcy_pressure(2 * dim + 1);
      const FEValuesExtractors::Vector displacement(2 * dim + 2);
      const FEValuesExtractors::Scalar l_multiplier(3 * dim + 2);

      std::vector<Tensor<2, dim>> stokes_grad_phi_u(stokes_dofs_per_cell);
      std::vector<Tensor<1, dim>> stokes_phi_u(stokes_dofs_per_cell);
      std::vector<double> stokes_div_phi_u(stokes_dofs_per_cell);
      std::vector<double> stokes_phi_p(stokes_dofs_per_cell);

      std::vector<Tensor<1, dim>> darcy_phi_u(biot_dofs_per_cell);
      std::vector<double> lm_phi(biot_dofs_per_cell);
      std::vector<Tensor<1, dim>> disp_phi_u(biot_dofs_per_cell);
      std::vector<double> darcy_phi_p(biot_dofs_per_cell);

      Vector<double> local_old_solid_solid_BJS_vector(biot_dofs_per_cell);
      Vector<double> local_old_fluid_solid_BJS_vector(stokes_dofs_per_cell);
      Vector<double> local_old_solid_interface_vector(biot_dofs_per_cell);

      std::vector<types::global_dof_index> neighbor_fluid_dof_indices(stokes_dofs_per_cell);

      const unsigned int n_face_quadrature_points = stokes_fe_face_values.n_quadrature_points;
      std::vector<double> darcy_bc_values(n_face_quadrature_points);
      std::vector<double> stokes_bc_values(n_face_quadrature_points);


      typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();

      for (; cell != endc; ++cell)
        {
        hp_fe_values.reinit(cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();


        local_rhs.reinit(cell->get_fe().dofs_per_cell);

        local_dof_indices.resize(cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        // assemble fluid volume terms
        if (cell_is_in_fluid_domain(cell))
          {
          //std::cout << "############ CELL [" << cell->center() << "] ############" << std::endl;

          const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
          //    std::vector<Vector<double>> rhs_stokes_values(fe_values.n_quadrature_points, Vector<double>(dim));
          //    std::vector<double>     src_stokes_values(fe_values.n_quadrature_points);
          std::vector<Tensor<1, dim>> old_stokes_velocity_values(fe_values.n_quadrature_points);

          //    right_hand_side_stokes.vector_value_list(fe_values.get_quadrature_points(), rhs_stokes_values);
          //    source_stokes.value_list(fe_values.get_quadrature_points(), src_stokes_values);

          fe_values[stokes_velocities].get_function_values(old_solution, old_stokes_velocity_values);

          Assert (dofs_per_cell == stokes_dofs_per_cell, ExcInternalError());

          const PointHistory<dim> *local_quadrature_points_data = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());


          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
            const Tensor<1, dim> &old_stokes_velocity = local_quadrature_points_data[q].old_stokes_velocity;

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
              stokes_phi_u[k] = fe_values[stokes_velocities].value(k, q);
              stokes_phi_p[k] = fe_values[stokes_pressure].value(k, q);

              local_rhs(k) += ((rho_f / time_step) * old_stokes_velocity * stokes_phi_u[k])
                              //   + rhs_stokes_values[q][0] * stokes_phi_u[k][0]
                              //     + rhs_stokes_values[q][1] * stokes_phi_u[k][1]
                              //     + src_stokes_values[q] * stokes_phi_p[k])
                              * fe_values.JxW(q);

              //std::cout << "  !!!! -- Fluid domain (volume), quad: " << q << " : " << old_stokes_velocity << std::endl;
              }

            }

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) && cell->face(f)->boundary_id() == 1)
              {
              stokes_fe_face_values.reinit(cell, f);
              stokes_pressure_bc.value_list(stokes_fe_face_values.get_quadrature_points(), stokes_bc_values);

              for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
                {
                const Tensor<1, dim> normal_vector = stokes_fe_face_values.normal_vector(q);
                for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                  {
                  local_rhs(i) += -(stokes_fe_face_values[stokes_velocities].value(i, q) * normal_vector
                                    * stokes_bc_values[q]
                                    * stokes_fe_face_values.JxW(q));
                  //std::cout << "  !!!! -- Fluid domain (boundary): " << local_rhs(i) << " " << stokes_bc_values[q] << std::endl;
                  }

                }
              }
          } else
          // assemble solid volume terms
          {
          const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;

          //  std::vector<double> rhs_biot_values(fe_values.n_quadrature_points);
          //  std::vector<Vector<double>> rhs_darcy_values(fe_values.n_quadrature_points, Vector<double>(dim));
          std::vector<double> old_pressure_values(fe_values.n_quadrature_points);
          std::vector<Vector<double>> old_disp_values(fe_values.n_quadrature_points, Vector<double>(dim));
          std::vector<double> old_disp_div_values(fe_values.n_quadrature_points);

          Assert (dofs_per_cell == biot_dofs_per_cell, ExcInternalError());

          //  right_hand_side_biot.value_list(fe_values.get_quadrature_points(), rhs_biot_values);
          //   right_hand_side_darcy.vector_value_list(fe_values.get_quadrature_points(), rhs_darcy_values);

          fe_values[darcy_pressure].get_function_values(old_solution, old_pressure_values);
          fe_values[displacement].get_function_divergences(old_solution, old_disp_div_values);

          const PointHistory<dim> *local_quadrature_points_data = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
            const double &old_darcy_pressure = local_quadrature_points_data[q].old_darcy_pressure;
            const double &old_displacement_div = local_quadrature_points_data[q].old_displacement_div;
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
              darcy_phi_p[k] = fe_values[darcy_pressure].value(k, q);
              disp_phi_u[k] = fe_values[displacement].value(k, q);

              local_rhs(k) += ((1 / time_step) * s_0 * old_darcy_pressure * darcy_phi_p[k]
                               + (1 / time_step) * alpha_p * old_displacement_div * darcy_phi_p[k])
                              //  + rhs_biot_values[q] * darcy_phi_p[k]
                              //  + rhs_darcy_values[q][0] * disp_phi_u[k][0]
                              //  + rhs_darcy_values[q][1] * disp_phi_u[k][1])
                              * fe_values.JxW(q);

              //std::cout << "  !!!! -- Solid domain (volume): " << local_rhs(k) << std::endl;
              }
            }

          
         for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
           if (cell->at_boundary(f))  //&& cell->face(f)->boundary_id() == 5
           {
             biot_fe_face_values.reinit(cell, f);
             darcy_pressure_bc.value_list(biot_fe_face_values.get_quadrature_points(), darcy_bc_values);

             for (unsigned int q = 0; q < n_face_quadrature_points; ++q) 
             {
               const Tensor<1, dim> normal_vector = biot_fe_face_values.normal_vector(q);
               for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                 local_rhs(i) += -(biot_fe_face_values[darcy_velocities].value(i, q) * normal_vector
                    * darcy_bc_values[q]
                    * biot_fe_face_values.JxW(q));
              }
            }
          }

        //std::cout << "Second place: " << system_rhs.l2_norm() << std::endl;

        // put into global rhs vector
        for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
          {
            system_rhs(local_dof_indices[i]) += local_rhs(i);
          }

        //std::cout << "Third place: " << system_rhs.l2_norm() << std::endl;

        // assemble old interface terms   REMOVE FALSE FROM HERE WHEN STOPPED DEBUGGING
        if (cell_is_in_solid_domain(cell))
          {
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) == false)
              {
              // if neighbor is in fluid domain and is on the same level of refinement
              // and doesn't have children
              if ((cell->neighbor(f)->level() == cell->level())
                  && (cell->neighbor(f)->has_children() == false)
                  && (cell_is_in_fluid_domain(cell->neighbor(f))))
                {
                biot_fe_face_values.reinit(cell, f);
                stokes_fe_face_values.reinit(cell->neighbor(f), cell->neighbor_of_neighbor(f));

                cell->neighbor(f)->get_dof_indices(neighbor_fluid_dof_indices);

                //std::vector<Tensor<1, dim>> old_disp_values(fe_values.n_quadrature_points);
                std::vector<Tensor<1, dim>> old_disp_values(biot_fe_face_values.n_quadrature_points);

                const PointHistory<dim> *local_quadrature_points_data = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

                // b-gamma term with displacement time derivative
                assemble_old_solid_interface_terms(f, biot_fe_face_values, disp_phi_u,
                                                   lm_phi, local_quadrature_points_data,
                                                   local_old_solid_interface_vector,
                                                   local_old_solid_solid_BJS_vector);
                // fluid-solid part of bjs term with displacement time derivative
                assemble_old_fluid_solid_BJS_term(f, biot_fe_face_values, stokes_fe_face_values,
                                                  stokes_phi_u, local_quadrature_points_data,
                                                  local_old_fluid_solid_BJS_vector);


                for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                  {
                  system_rhs(local_dof_indices[i]) += local_old_solid_interface_vector(i);
                  system_rhs(local_dof_indices[i]) += local_old_solid_solid_BJS_vector(i);
                  }
                for (unsigned int i = 0; i < cell->neighbor(f)->get_fe().dofs_per_cell; ++i)
                  system_rhs(neighbor_fluid_dof_indices[i]) += local_old_fluid_solid_BJS_vector(i);
                } else if ((cell->neighbor(f)->level() == cell->level())
                           && (cell->neighbor(f)->has_children() == true))
                {
                // if neighbor is in fluid domain and has children
                for (unsigned int subface = 0; subface < cell->face(f)->n_children(); ++subface)
                  if (cell_is_in_fluid_domain(cell->neighbor_child_on_subface(f, subface)))
                    {
                    biot_fe_subface_values.reinit(cell, f, subface);
                    stokes_fe_face_values.reinit(cell->neighbor_child_on_subface(f, subface),
                                                 cell->neighbor_of_neighbor(f));

                    cell->neighbor(f)->get_dof_indices(neighbor_fluid_dof_indices);

                    //std::vector<Tensor<1, dim>> old_disp_values(fe_values.n_quadrature_points);
                    std::vector<Tensor<1, dim>> old_disp_values(biot_fe_face_values.n_quadrature_points);

                    const PointHistory<dim> *local_quadrature_points_data = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());


                    // b-gamma term with displacement time derivative
                    assemble_old_solid_interface_terms(f, biot_fe_face_values, disp_phi_u,
                                                       lm_phi, local_quadrature_points_data,
                                                       local_old_solid_interface_vector,
                                                       local_old_solid_solid_BJS_vector);
                    // fluid-solid part of bjs term with displacement time derivative
                    assemble_old_fluid_solid_BJS_term(f, biot_fe_subface_values, stokes_fe_face_values,
                                                      stokes_phi_u, local_quadrature_points_data,
                                                      local_old_fluid_solid_BJS_vector);


                    for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                      {
                      system_rhs(local_dof_indices[i]) += local_old_solid_interface_vector(i);
                      system_rhs(local_dof_indices[i]) += local_old_solid_solid_BJS_vector(i);
                      }
                    for (unsigned int i = 0;
                         i < cell->neighbor_child_on_subface(f, subface)->get_fe().dofs_per_cell; ++i)
                      system_rhs(neighbor_fluid_dof_indices[i]) += local_old_fluid_solid_BJS_vector(i);
                    }
                } else if (cell->neighbor_is_coarser(f)
                           && cell_is_in_fluid_domain(cell->neighbor(f)))
                {
                // if neighbor is in fluid domain and is coarser
                biot_fe_face_values.reinit(cell, f);
                stokes_fe_subface_values.reinit(cell->neighbor(f),
                                                cell->neighbor_of_coarser_neighbor(f).first,
                                                cell->neighbor_of_coarser_neighbor(f).second);

                cell->neighbor(f)->get_dof_indices(neighbor_fluid_dof_indices);

                //std::vector<Tensor<1, dim>> old_disp_values(fe_values.n_quadrature_points);
                std::vector<Tensor<1, dim>> old_disp_values(biot_fe_face_values.n_quadrature_points);

                const PointHistory<dim> *local_quadrature_points_data = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

                // b-gamma term with displacement time derivative
                assemble_old_solid_interface_terms(f, biot_fe_face_values, disp_phi_u,
                                                   lm_phi, local_quadrature_points_data,
                                                   local_old_solid_interface_vector,
                                                   local_old_solid_solid_BJS_vector);
                // fluid-solid part of bjs term with displacement time derivative
                assemble_old_fluid_solid_BJS_term(f, biot_fe_face_values, stokes_fe_subface_values,
                                                  stokes_phi_u, local_quadrature_points_data,
                                                  local_old_fluid_solid_BJS_vector);


                for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                  {
                  system_rhs(local_dof_indices[i]) += local_old_solid_interface_vector(i);
                  system_rhs(local_dof_indices[i]) += local_old_solid_solid_BJS_vector(i);
                  }
                for (unsigned int i = 0; i < cell->neighbor(f)->get_fe().dofs_per_cell; ++i)
                  system_rhs(neighbor_fluid_dof_indices[i]) += local_old_fluid_solid_BJS_vector(i);
                }
              }
          }
        }
      }


    template<int dim>
    void FluidStructureProblem<dim>::assemble_harmonic_extension_rhs()
      {
      TimerOutput::Scope t(computing_timer, "Assemble Harmonic Extension RHS");
      harmonic_rhs = 0;

      hp::FEValues<dim> hp_fe_values(fe_harmonic, q_collection_harmonic,
                                     update_values |
                                     update_quadrature_points |
                                     update_JxW_values |
                                     update_gradients);

      hp::FEValues<dim> true_fe_values(fe_collection, q_collection,
                                       update_values);

      const QGauss<dim - 1> common_face_quadrature(elasticity_degree + 2);

      FEFaceValues<dim> fe_face_values(harmonic_extension_fe_fluid,
                                       common_face_quadrature,
                                       update_values |
                                       update_JxW_values |
                                       update_quadrature_points |
                                       update_normal_vectors);
      FEFaceValues<dim> biot_fe_face_values(biot_fe,
                                            common_face_quadrature,
                                            update_values |
                                            update_JxW_values |
                                            update_quadrature_points |
                                            update_normal_vectors);

      FESubfaceValues<dim> fe_subface_values(harmonic_extension_fe_fluid,
                                             common_face_quadrature,
                                             update_JxW_values |
                                             update_normal_vectors);
      FESubfaceValues<dim> biot_fe_subface_values(biot_fe,
                                                  common_face_quadrature,
                                                  update_values);


      const unsigned int dofs_per_cell_h = harmonic_extension_fe_fluid.dofs_per_cell;
      const unsigned int biot_dofs_per_cell = biot_fe.dofs_per_cell;


      Vector<double> local_interface_vector;

      std::vector<types::global_dof_index> local_dof_indices;

//      const FEValuesExtractors::Vector disp(0);
//      const FEValuesExtractors::Vector lm(dim);

      //const FEValuesExtractors::Vector displacement(2 * dim + 2);

      std::vector<Tensor<1, dim> > mu_phi(dofs_per_cell_h);

      std::vector<types::global_dof_index> neighbor_solid_dof_indices(biot_dofs_per_cell);

      //const unsigned int n_face_quadrature_points = fe_face_values.n_quadrature_points;

      typename hp::DoFHandler<dim>::active_cell_iterator
              cell = dof_handler_harmonic.begin_active(),
              endc = dof_handler_harmonic.end();

      for (; cell != endc; ++cell)
        {
        hp_fe_values.reinit(cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();


        local_interface_vector.reinit(cell->get_fe().dofs_per_cell);

        local_dof_indices.resize(cell->get_fe().dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        // assemble interface terms
        if (cell_is_in_fluid_domain(cell))
          {
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) == false)
              {
              // if neighbor is in fluid domain and is on the same level of refinement
              // and doesn't have children
              if ((cell->neighbor(f)->level() == cell->level())
                  && (cell->neighbor(f)->has_children() == false)
                  && (cell_is_in_solid_domain(cell->neighbor(f))))
                {
                fe_face_values.reinit(cell, f);
                typename hp::DoFHandler<dim>::active_cell_iterator data_cell(&triangulation, cell->level(),
                                                                             cell->index(),
                                                                             &dof_handler);
                biot_fe_face_values.reinit(data_cell->neighbor(f), data_cell->neighbor_of_neighbor(f));
                true_fe_values.reinit(data_cell->neighbor(f));

                std::vector<Tensor<1, dim>> displacement_data(fe_values.n_quadrature_points);

                assemble_interface_rhs_terms_harmonic_extension(true_fe_values, fe_face_values,
                                                                mu_phi, displacement_data,
                                                                local_interface_vector);

                for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                  harmonic_rhs(local_dof_indices[i]) += local_interface_vector(i);
                } else if ((cell->neighbor(f)->level() == cell->level())
                           && (cell->neighbor(f)->has_children() == true))
                {
                // if neighbor is in fluid domain and has children
                for (unsigned int subface = 0; subface < cell->face(f)->n_children(); ++subface)
                  if (cell_is_in_solid_domain(cell->neighbor_child_on_subface(f, subface)))
                    {
                    fe_subface_values.reinit(cell, f, subface);
                    typename hp::DoFHandler<dim>::active_cell_iterator data_cell(&triangulation, cell->level(),
                                                                                 cell->index(),
                                                                                 &dof_handler);
                    biot_fe_face_values.reinit(data_cell->neighbor_child_on_subface(f, subface),
                                               data_cell->neighbor_of_neighbor(f));
                    true_fe_values.reinit(data_cell->neighbor_child_on_subface(f, subface));

                    std::vector<Tensor<1, dim>> displacement_data(fe_values.n_quadrature_points);

                    assemble_interface_rhs_terms_harmonic_extension(true_fe_values, fe_subface_values,
                                                                    mu_phi, displacement_data,
                                                                    local_interface_vector);

                    for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                      harmonic_rhs(local_dof_indices[i]) += local_interface_vector(i);
                    }
                } else if (cell->neighbor_is_coarser(f)
                           && cell_is_in_solid_domain(cell->neighbor(f)))
                {
                // if neighbor is in fluid domain and is coarser
                fe_face_values.reinit(cell, f);
                typename hp::DoFHandler<dim>::active_cell_iterator data_cell(&triangulation, cell->level(),
                                                                             cell->index(),
                                                                             &dof_handler);
                biot_fe_subface_values.reinit(data_cell->neighbor(f),
                                              data_cell->neighbor_of_coarser_neighbor(f).first,
                                              data_cell->neighbor_of_coarser_neighbor(f).second);

                true_fe_values.reinit(data_cell->neighbor(f));

                std::vector<Tensor<1, dim>> displacement_data(fe_values.n_quadrature_points);

                assemble_interface_rhs_terms_harmonic_extension(true_fe_values, fe_face_values,
                                                                mu_phi, displacement_data,
                                                                local_interface_vector);

                for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                  harmonic_rhs(local_dof_indices[i]) += local_interface_vector(i);
                }
              }
          }
        }
      //  cout << "HARMONIC RHS NORM IS = " << harmonic_rhs.norm_sqr() << endl;
      }


    /*
   *  Assemble FLUID-FLUID and interface BJS TERM and
   *  there is no RHS part of it
   */
    template<int dim>
    void
    FluidStructureProblem<dim>::
    assemble_fluid_interface_terms(const FEFaceValuesBase<dim> &biot_fe_face_values,
                                   const FEFaceValuesBase<dim> &stokes_fe_face_values,
                                   std::vector<double> &lm_phi,
                                   std::vector<Tensor<1, dim> > &stokes_phi_u,
                                   FullMatrix<double> &local_fluid_interface_matrix,
                                   FullMatrix<double> &local_fluid_fluid_BJS_matrix) const
      {
      Assert (stokes_fe_face_values.n_quadrature_points ==
              biot_fe_face_values.n_quadrature_points,
              ExcInternalError());

      const unsigned int n_face_quadrature_points
              = biot_fe_face_values.n_quadrature_points;

      const FEValuesExtractors::Vector stokes_velocities(0);
      const FEValuesExtractors::Scalar l_multiplier(3 * dim + 2);

      local_fluid_interface_matrix = 0;
      local_fluid_fluid_BJS_matrix = 0;
      for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
        {
        const Tensor<1, dim> normal_vector = stokes_fe_face_values.normal_vector(q);

        // Evaluate Stokes velocity test functions
        for (unsigned int k = 0; k < stokes_fe_face_values.dofs_per_cell; ++k)
          stokes_phi_u[k] = stokes_fe_face_values[stokes_velocities].value(k, q);

        // Evaluate LM test functions
        for (unsigned int k = 0; k < biot_fe_face_values.dofs_per_cell; ++k)
          lm_phi[k] = biot_fe_face_values[l_multiplier].value(k, q);

        for (unsigned int i = 0; i < stokes_fe_face_values.dofs_per_cell; ++i)
          {
          // signs are ok, no dt at all (NEED TO ADD THE PERMEABITY LATER ON)
          for (unsigned int j = 0; j < stokes_fe_face_values.dofs_per_cell; ++j)      // this on diagonal
            local_fluid_fluid_BJS_matrix(i, j) += alpha_bjs *            // < u . tau, v . tau >
                                                  ((stokes_phi_u[i] * stokes_phi_u[j])
                                                   -
                                                   (stokes_phi_u[i] * normal_vector) *
                                                   (stokes_phi_u[j] * normal_vector))
                                                  * stokes_fe_face_values.JxW(q);

          for (unsigned int j = 0;
               j < biot_fe_face_values.dofs_per_cell; ++j)        // this should go above the diagonal
            local_fluid_interface_matrix(i, j) += (stokes_phi_u[i] * normal_vector) * // < v . n, lambda . n >
                                                  lm_phi[j] * stokes_fe_face_values.JxW(q);
          }
        }
      }


    /*
   *  Assemble FLUID-SOLID BJS TERM and
   *  the RHS part of it due
   *  to time-discretization
   */
    template<int dim>
    void
    FluidStructureProblem<dim>::
    assemble_fluid_solid_BJS_term(const FEFaceValuesBase<dim> &biot_fe_face_values,
                                  const FEFaceValuesBase<dim> &stokes_fe_face_values,
                                  std::vector<Tensor<1, dim>> &stokes_phi_u,
                                  std::vector<Tensor<1, dim>> &disp_phi_u,
                                  FullMatrix<double> &local_fluid_solid_BJS_matrix) const
      {
      const unsigned int n_face_quadrature_points
              = stokes_fe_face_values.n_quadrature_points;

      const FEValuesExtractors::Vector stokes_velocities(0);
      const FEValuesExtractors::Vector displacement(2 * dim + 2);

      local_fluid_solid_BJS_matrix = 0;
      for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
        {
        const Tensor<1, dim> normal_vector = stokes_fe_face_values.normal_vector(q);

        for (unsigned int k = 0; k < stokes_fe_face_values.dofs_per_cell; ++k)
          stokes_phi_u[k] = stokes_fe_face_values[stokes_velocities].value(k, q);

        for (unsigned int k = 0; k < biot_fe_face_values.dofs_per_cell; ++k)
          disp_phi_u[k] = biot_fe_face_values[displacement].value(k, q);

        for (unsigned int i = 0; i < stokes_fe_face_values.dofs_per_cell; ++i)    // this goes above diagonal
          for (unsigned int j = 0; j < biot_fe_face_values.dofs_per_cell; ++j)    // < v . tau, dt eta . tau >
            local_fluid_solid_BJS_matrix(i, j) += -(1.0 / time_step) * alpha_bjs *
                                                  ((stokes_phi_u[i] * disp_phi_u[j]) - (stokes_phi_u[i] * normal_vector)
                                                                                       * (disp_phi_u[j] * normal_vector)) * stokes_fe_face_values.JxW(q);

        }
      }

    template<int dim>
    void
    FluidStructureProblem<dim>::
    assemble_old_fluid_solid_BJS_term(const int faceno,
                                      const FEFaceValuesBase<dim> &biot_fe_face_values,
                                      const FEFaceValuesBase<dim> &stokes_fe_face_values,
                                      std::vector<Tensor<1, dim> > &stokes_phi_u,
                                      const PointHistory<dim> *local_quadrature_points_data,
                                      Vector<double> &local_old_fluid_solid_BJS_vector) const
      {
      const unsigned int n_face_quadrature_points
              = stokes_fe_face_values.n_quadrature_points;

      const unsigned int n_q_points = q_collection[1].size();
      const FEValuesExtractors::Vector stokes_velocities(0);

      local_old_fluid_solid_BJS_vector = 0;
      for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
        {
        const Tensor<1, dim> normal_vector = stokes_fe_face_values.normal_vector(q);
        const Tensor<1, dim> &old_displacement = local_quadrature_points_data[q + n_q_points + faceno *
                                                                                               n_face_quadrature_points].old_displacement_face;

        for (unsigned int k = 0; k < stokes_fe_face_values.dofs_per_cell; ++k)
          stokes_phi_u[k] = stokes_fe_face_values[stokes_velocities].value(k, q);

        for (unsigned int i = 0; i < stokes_fe_face_values.dofs_per_cell; ++i)
          local_old_fluid_solid_BJS_vector(i) +=
                  -alpha_bjs * (1 / time_step) * ((stokes_phi_u[i] * old_displacement)
                                                  - (stokes_phi_u[i] * normal_vector) *
                                                    (old_displacement * normal_vector)) *
                  stokes_fe_face_values.JxW(q);

        }
      }
    // done fluid-solid

    /*
   *  Assemble SOLID-SOLID interface and BJS TERM and
   *  the RHS part of it due
   *  to time-discretization
   */
    template<int dim>
    void
    FluidStructureProblem<dim>::
    assemble_solid_interface_terms(const FEFaceValuesBase<dim> &biot_fe_face_values,
                                   const FEFaceValuesBase<dim> &stokes_fe_face_values,
                                   std::vector<double> &lm_phi,
                                   std::vector<Tensor<1, dim>> &darcy_phi_u,
                                   std::vector<Tensor<1, dim>> &disp_phi_u,
                                   FullMatrix<double> &local_solid_interface_matrix,
                                   FullMatrix<double> &local_solid_solid_BJS_matrix) const
      {
      const unsigned int n_face_quadrature_points = biot_fe_face_values.n_quadrature_points;

      const FEValuesExtractors::Vector darcy_velocities(dim + 1);
      const FEValuesExtractors::Vector displacement(2 * dim + 2);
      const FEValuesExtractors::Scalar l_multiplier(3 * dim + 2);

      local_solid_interface_matrix = 0;
      local_solid_solid_BJS_matrix = 0;
      for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
        {
        const Tensor<1, dim> normal_vector = biot_fe_face_values.normal_vector(q);
        const Tensor<1, dim> normal_vector_fluid = stokes_fe_face_values.normal_vector(q);
        for (unsigned int k = 0; k < biot_fe_face_values.dofs_per_cell; ++k)
          {
          darcy_phi_u[k] = biot_fe_face_values[darcy_velocities].value(k, q);
          lm_phi[k] = biot_fe_face_values[l_multiplier].value(k, q);
          disp_phi_u[k] = biot_fe_face_values[displacement].value(k, q);
          }

        for (unsigned int i = 0; i < biot_fe_face_values.dofs_per_cell; ++i)
          {
          for (unsigned int j = 0; j < biot_fe_face_values.dofs_per_cell; ++j)
            {
            local_solid_solid_BJS_matrix(i, j) += (1 / time_step) * alpha_bjs
                                                  * ((disp_phi_u[i] * disp_phi_u[j]) -
                                                     (disp_phi_u[i] * normal_vector_fluid) *
                                                     (disp_phi_u[j] * normal_vector_fluid))
                                                  * biot_fe_face_values.JxW(q);

            local_solid_interface_matrix(i, j) +=
                    ((darcy_phi_u[i] * normal_vector) * lm_phi[j]     // v_p dot n_p , lambda
                     + (darcy_phi_u[j] * normal_vector) * lm_phi[i]   // u_p dot n_p , mu
                     + (disp_phi_u[i] * normal_vector) * lm_phi[j]   // ksi dot n_p , lambda
                     + (1 / time_step) * (disp_phi_u[j] * normal_vector)                // dt eta dot n_p, mu
                       * lm_phi[i]) * biot_fe_face_values.JxW(q);
            }
          }
        }
      }

    template<int dim>
    void
    FluidStructureProblem<dim>::
    assemble_old_solid_interface_terms(const int faceno,
                                       const FEFaceValuesBase<dim> &biot_fe_face_values,
                                       std::vector<Tensor<1, dim> > &disp_phi_u,
                                       std::vector<double> &lm_phi,
                                       const PointHistory<dim> *local_quadrature_points_data,
                                       Vector<double> &local_old_solid_interface_vector,
                                       Vector<double> &local_old_solid_solid_BJS_vector) const
      {
      const unsigned int n_face_quadrature_points
              = biot_fe_face_values.n_quadrature_points;

      const FEValuesExtractors::Vector displacement(2 * dim + 2);
      const FEValuesExtractors::Scalar l_multiplier(3 * dim + 2);

      const unsigned int n_q_points = q_collection[1].size();

      local_old_solid_interface_vector = 0;
      local_old_solid_solid_BJS_vector = 0;

      for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
        {
        const Tensor<1, dim> normal_vector = biot_fe_face_values.normal_vector(q);
        const Tensor<1, dim> &old_displacement = local_quadrature_points_data[q + n_q_points + faceno *
                                                                                               n_face_quadrature_points].old_displacement_face;

        for (unsigned int k = 0; k < biot_fe_face_values.dofs_per_cell; ++k)
          {
          lm_phi[k] = biot_fe_face_values[l_multiplier].value(k, q);
          disp_phi_u[k] = biot_fe_face_values[displacement].value(k, q);
          }

        for (unsigned int i = 0; i < biot_fe_face_values.dofs_per_cell; ++i)
          {
          local_old_solid_interface_vector(i) += ((1 / time_step) * (old_displacement * normal_vector)
                                                  * lm_phi[i]) * biot_fe_face_values.JxW(q);

          local_old_solid_solid_BJS_vector(i) += (1 / time_step) * alpha_bjs * ((disp_phi_u[i] * old_displacement)
                                                                                - (disp_phi_u[i] * normal_vector) *
                                                                                  (old_displacement *
                                                                                   normal_vector)) *
                                                 biot_fe_face_values.JxW(q);
          }
        }
      }

    template<int dim>
    void
    FluidStructureProblem<dim>::
    assemble_interface_rhs_terms_harmonic_extension(const hp::FEValues<dim> &true_fe_values,
                                                    const FEFaceValuesBase<dim> &harmonic_fe_face_values,
                                                    std::vector<Tensor<1, dim>> &mu_phi,
                                                    std::vector<Tensor<1, dim> > &disp_values,
                                                    Vector<double> &local_interface_vector) const
      {
      const unsigned int n_face_quadrature_points = harmonic_fe_face_values.n_quadrature_points;

      const FEValuesExtractors::Vector lm(dim);

      const FEValuesExtractors::Vector displacement(2 * dim + 2);

      const FEValues<dim> &fe_values = true_fe_values.get_present_fe_values();
      fe_values[displacement].get_function_values(solution, disp_values);


      local_interface_vector = 0;
      for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
        {

        for (unsigned int k = 0; k < harmonic_fe_face_values.dofs_per_cell; ++k)
          {
          mu_phi[k] = harmonic_fe_face_values[lm].value(k, q);
          local_interface_vector(k) += (mu_phi[k] * disp_values[q]) * harmonic_fe_face_values.JxW(q);
          }

        }
      }


    template<int dim>
    void
    FluidStructureProblem<dim>::solve()
      {
      TimerOutput::Scope t(computing_timer, "Solve");

      SparseDirectUMFPACK direct_solver;
      Vector<double> difference;
      difference.reinit(dof_handler.n_dofs());
      previous_iteration_solution.reinit(dof_handler.n_dofs());

      unsigned int max_iter = 25;
      unsigned int iter = 0;
      double tol = 1e-5;
      double error = 1;

      assemble_rhs();

      /////////////////////

      const FEValuesExtractors::Scalar displacement_tangential_z(2*dim+4);
      cout << " Interpolating Elasticity BC ..." << endl;
      VectorTools::interpolate_boundary_values(dof_handler,
                                               5,
                                               ElasticityBoundaryValues<dim>(time),
                                               boundary_values,
                                               fe_collection.component_mask(displacement_tangential_z));


      /////////////////////


      while (error > tol && iter < max_iter)
        {
        if (iter == 0)
          previous_iteration_solution = old_solution;

        system_matrix.reinit(sparsity_pattern);
        assemble_system();

        MatrixTools::apply_boundary_values(boundary_values,
                                           system_matrix, solution, system_rhs);

        direct_solver.initialize(system_matrix);
        direct_solver.vmult(solution, system_rhs);

        for (unsigned int i = 0; i < solution.size(); ++i)
          difference[i] = solution[i] - previous_iteration_solution[i];

        if (previous_iteration_solution.l2_norm() != 0)
          error = difference.l2_norm() / previous_iteration_solution.l2_norm();
        else
          error = difference.l2_norm();

        previous_iteration_solution = solution;
        iter++;
        }

      std::cout << "\n  **** # of iterations: " << iter << ", residual: " << error << std::endl;
      }


    template<int dim>
    void
    FluidStructureProblem<dim>::solve_harmonic_extension()
      {
      TimerOutput::Scope t(computing_timer, "Solve Harmonic Extension");
      SparseDirectUMFPACK direct_solver;

      direct_solver.initialize(system_matrix_harmonic);
      direct_solver.vmult(harmonic_extension, harmonic_rhs);
      }

    template<int dim>
    void FluidStructureProblem<dim>::move_mesh()
      {
      TimerOutput::Scope t(computing_timer, "Move mesh");
      cout << "    Moving mesh..." << endl;
      std::vector<bool> vertex_touched(triangulation.n_vertices(),
                                       false);
      //cout << " L inf norm of solution is: " << solution.linfty_norm() << endl;
      for (typename hp::DoFHandler<dim>::active_cell_iterator
                   cell = dof_handler.begin_active();
           cell != dof_handler.end(); ++cell)
        {
        if (cell_is_in_solid_domain(cell))
          {
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
            {
            if (vertex_touched[cell->vertex_index(v)] == false)
              {
              vertex_touched[cell->vertex_index(v)] = true;
              Point<dim> vertex_displacement;
              for (unsigned int d = 0; d < dim; ++d)
                vertex_displacement[d] =
                        solution(cell->vertex_dof_index(v, d, 1)) - old_solution(cell->vertex_dof_index(v, d, 1));

              cell->vertex(v) += vertex_displacement;
              }
            }
          } else
          {
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
            {
            if (vertex_touched[cell->vertex_index(v)] == false)
              {
              vertex_touched[cell->vertex_index(v)] = true;
              Point<dim> vertex_displacement;
              typename hp::DoFHandler<dim>::active_cell_iterator data_cell(&triangulation, cell->level(),
                                                                           cell->index(),
                                                                           &dof_handler_harmonic);

              for (unsigned int d = 0; d < dim; ++d)
                vertex_displacement[d] = harmonic_extension(data_cell->vertex_dof_index(v, d, 0)) -
                                         old_harmonic_extension(data_cell->vertex_dof_index(v, d, 0));

              //if( fabs(vertex_displacement[0])>1e-15 || fabs(vertex_displacement[1])>1e-15)
              //  cout << "MOVEMENT IN FLUID REGION: " << vertex_displacement[0] << " , " << vertex_displacement[1]<< endl;

              cell->vertex(v) += vertex_displacement;
              }
            }
          }
        }
      }

    template<int dim>
    void FluidStructureProblem<dim>::setup_quadrature_point_history()
      {
      TimerOutput::Scope t(computing_timer, "Setup QPH");
      unsigned int f_cells = 0, s_cells = 0;
      for (typename hp::DoFHandler<dim>::active_cell_iterator
                   cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
        {
        if (cell_is_in_fluid_domain(cell))
          ++f_cells;
        else
          ++s_cells;
        }

      const QGauss<dim - 1> common_face_quadrature(std::max(std::max(stokes_degree + 2,
                                                                     darcy_degree + 2), elasticity_degree + 2));

      const unsigned int q_p_h_size =
              f_cells * (q_collection[0].size() + GeometryInfo<dim>::faces_per_cell * common_face_quadrature.size())
              + s_cells * (q_collection[1].size() + GeometryInfo<dim>::faces_per_cell * common_face_quadrature.size());

      triangulation.clear_user_data();
      {
        std::vector<PointHistory<dim> > tmp;
        tmp.swap(quadrature_point_history);
      }

      quadrature_point_history.resize(q_p_h_size);
      unsigned int history_index = 0;
      for (typename Triangulation<dim>::active_cell_iterator
                   cell = triangulation.begin_active();
           cell != triangulation.end(); ++cell)
        {
        cell->set_user_pointer(&quadrature_point_history[history_index]);
        if (cell_is_in_fluid_domain(cell))
          history_index += q_collection[0].size() + GeometryInfo<dim>::faces_per_cell * common_face_quadrature.size();
        else
          history_index += q_collection[1].size() + GeometryInfo<dim>::faces_per_cell * common_face_quadrature.size();
        }

      Assert (history_index == quadrature_point_history.size(), ExcInternalError());
      }

    template<int dim>
    void FluidStructureProblem<dim>::update_quadrature_point_history()
      {
      TimerOutput::Scope t(computing_timer, "Update QPH");
      hp::FEValues<dim> hp_fe_values(fe_collection, q_collection,
                                     update_values |
                                     update_quadrature_points |
                                     update_JxW_values |
                                     update_gradients);

      const QGauss<dim - 1> common_face_quadrature(std::max(std::max(stokes_degree + 2,
                                                                     darcy_degree + 2), elasticity_degree + 2));

      FEFaceValues<dim> biot_fe_face_values(biot_fe,
                                            common_face_quadrature,
                                            update_values |
                                            update_JxW_values |
                                            update_quadrature_points |
                                            update_normal_vectors |
                                            update_gradients);

      const FEValuesExtractors::Vector stokes_velocities(0);
      const FEValuesExtractors::Scalar darcy_pressure(2 * dim + 1);
      const FEValuesExtractors::Vector displacement(2 * dim + 2);


      for (typename hp::DoFHandler<dim>::active_cell_iterator
                   cell = dof_handler.begin_active();
           cell != dof_handler.end(); ++cell)
        {
        hp_fe_values.reinit(cell);
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();


        std::vector<Tensor<1, dim> > stokes_vel(fe_values.n_quadrature_points);
        std::vector<Tensor<1, dim> > disp(fe_values.n_quadrature_points);
        std::vector<double> disp_div(fe_values.n_quadrature_points);
        std::vector<double> darcy_pr(fe_values.n_quadrature_points);
        PointHistory<dim> *local_quadrature_points_history
                = reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
        Assert (local_quadrature_points_history >=
                &quadrature_point_history.front(),
                ExcInternalError());
        Assert (local_quadrature_points_history <
                &quadrature_point_history.back(),
                ExcInternalError());

        fe_values[stokes_velocities].get_function_values(solution, stokes_vel);
        fe_values[displacement].get_function_values(solution, disp);
        fe_values[displacement].get_function_divergences(solution, disp_div);
        fe_values[darcy_pressure].get_function_values(solution, darcy_pr);

        unsigned int n_q_points =
                (cell_is_in_fluid_domain(cell) ? q_collection[0].size() : q_collection[1].size());

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
          local_quadrature_points_history[q].old_stokes_velocity = stokes_vel[q];
          local_quadrature_points_history[q].old_displacement_div = disp_div[q];
          local_quadrature_points_history[q].old_darcy_pressure = darcy_pr[q];
          }


        if (cell_is_in_solid_domain(cell))
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
            if (cell->at_boundary(f) == false)
              {
              // if neighbor is in fluid domain and is on the same level of refinement
              // and doesn't have children
              if ((cell->neighbor(f)->level() == cell->level())
                  && (cell->neighbor(f)->has_children() == false)
                  && (cell_is_in_fluid_domain(cell->neighbor(f))))
                {
                std::vector<Tensor<1, dim>> disp_face(biot_fe_face_values.n_quadrature_points);
                biot_fe_face_values.reinit(cell, f);
                biot_fe_face_values[displacement].get_function_values(solution, disp_face);

                for (unsigned int q = 0; q < biot_fe_face_values.n_quadrature_points; ++q)
                  local_quadrature_points_history[q + n_q_points + f *
                                                                   biot_fe_face_values.n_quadrature_points].old_displacement_face = disp_face[q];
                }
              }
            }


        }
      }


    template<int dim>
  void FluidStructureProblem<dim>::output_results(std::string solname, bool compute_diff)
      {
        //TimerOutput::Scope t(computing_timer, "Output results");

        // two files needed for proper visualization in paraview...
        std::vector<std::string> solution_names_f(dim, "velocity");
        solution_names_f.push_back("pressure");

        for (unsigned int d = 0; d < dim; ++d)
          solution_names_f.push_back("darcy_velocity");


        solution_names_f.push_back("darcy_pressure");

        for (unsigned int d = 0; d < dim; ++d)
          solution_names_f.push_back("displacement");


        solution_names_f.push_back("l_multiplier");

        //
        std::vector<std::string> solution_names_h(dim, "h_ext");
        for (unsigned int d = 0; d < dim; ++d)
          solution_names_h.push_back("lm");
        //
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation
                (dim, DataComponentInterpretation::component_is_part_of_vector);


        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);


        for (unsigned int d = 0; d < dim; ++d)
          data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);


        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        for (unsigned int d = 0; d < dim; ++d)
          data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);


        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation_h
                (dim, DataComponentInterpretation::component_is_part_of_vector);

        for (unsigned int d = 0; d < dim; ++d)
          data_component_interpretation_h.push_back(DataComponentInterpretation::component_is_part_of_vector);
        //
        DataOut<dim, hp::DoFHandler<dim> > data_out_f;
        int tmp = round(time / time_step);


//      Vector<double> soln(solution.size());
//      if (compute_diff && time > 1e-10)
//        {
//        int idx = tmp - 1;
//        //std::cout << "INDEX!!! : " << tmp-1 << ", TIME!!! : " << time << " " << time/time_step << std::endl;
//
//        //data_out_f.attach_dof_handler(dof_handler);
//
//        //      soln = solutions->at(idx);
//        for (unsigned int i = 0; i < soln.size(); ++i)
//          soln[i] = fabs(solutions->at(idx)[i] - solution[i]);
//
//        data_out_f.add_data_vector(dof_handler, soln, solution_names_f,
//                                   data_component_interpretation);
//
//        data_out_f.build_patches();
//
//        std::ostringstream filename_f;
//        filename_f << "difference_" << std::to_string(dim) << "d-" << std::to_string(tmp) << ".vtk";
//
//        std::ofstream output_f(filename_f.str().c_str());
//        data_out_f.write_vtk(output_f);
//        output_f.close();
//        }

        {
          Postprocessor<dim> postprocessor(this);

          data_out_f.clear();
          //data_out_f.attach_dof_handler(dof_handler);
          data_out_f.add_data_vector(dof_handler,
                                     solution,
                                     solution_names_f,
                                     data_component_interpretation);

          data_out_f.add_data_vector(dof_handler,
                                     solution,
                                     postprocessor);

          data_out_f.add_data_vector(dof_handler_harmonic,
                                     harmonic_extension,
                                     solution_names_h,
                                     data_component_interpretation_h);

          data_out_f.build_patches();

          std::ostringstream filename_f;
          filename_f << solname << "_" << std::to_string(dim) << "d-" << std::to_string(tmp) << ".vtk";
          std::ofstream output_f(filename_f.str().c_str());

          data_out_f.write_vtk(output_f);
          output_f.close();

          data_out_f.clear_input_data_references();
          data_out_f.clear();
        }
      }


    template<int dim>
    void FluidStructureProblem<dim>::run(const unsigned int refine, std::string solname, bool compute_diff)
      {
      make_grid();

      //GridTools::transform(&grid_transform<dim>, triangulation);
      for (unsigned int cycle = 0; cycle < refine; ++cycle)
        {
        setup_dofs();

        ConstraintMatrix constraints;
        constraints.close();

        VectorTools::project(dof_handler,
                             constraints,
                             q_collection,
                             InitialCondition<dim>(),
                             old_solution);

        harmonic_extension = 0.0;
        old_harmonic_extension = 0.0;
        solution = old_solution;
        previous_iteration_solution = 0.0;
        output_results();
        //cout << "Bandwidth: " << sparsity_pattern.bandwidth() << endl;

        for (unsigned int i = 0; i < num_time_steps; i++)
          {
          system_matrix_harmonic.reinit(sparsity_pattern_harmonic);

          time += time_step;
          cout << "###### t = " << time << " ######\n";

          std::cout << "   Solving..." << std::endl;
          solve();


          assemble_harmonic_extension_system();
          assemble_harmonic_extension_rhs();
          MatrixTools::apply_boundary_values(boundary_values_harmonic,
                                             system_matrix_harmonic, harmonic_extension, harmonic_rhs);


          solve_harmonic_extension();

          cout << "Solution norm: " << solution.l2_norm() << endl;
          cout << "    Updating quadrature point data..." << endl;
          update_quadrature_point_history();
          move_mesh();

          system_rhs = 0;
          harmonic_rhs = 0;

          std::cout << "   Writing output..." << std::endl;
          output_results(solname, compute_diff);
          old_solution = solution;

          cout << "Index when saving is: " << i << ", and time is " << time << std::endl;
          if (solutions != nullptr && !compute_diff)
            solutions->at(i) = solution;

          old_harmonic_extension = harmonic_extension;
          }
        time = 0.0;
        }

      computing_timer.print_summary();
      computing_timer.reset();

      cout << "Done." << endl;

      }

    template<int dim>
    void
    FluidStructureProblem<dim>::reset()
      {
      dof_handler.clear();
      dof_handler_harmonic.clear();
      triangulation.clear();
      }
  }


int main()
  {
  try
    {
    using namespace dealii;
    using namespace StokesBiot;

    MultithreadInfo::set_thread_limit();

    int n_tsteps = 60;
    std::vector<Vector<double>> solutions(n_tsteps);

    FluidStructureProblem<3> flow_problem(1, 0, 1, 0.0001, n_tsteps, &solutions);
    flow_problem.set_vmodel(1);
    flow_problem.run(1, "nonlinear");

//    std::cout << "Done first pass" << std::endl;
//    flow_problem.reset();
//    flow_problem.set_vmodel(0);
//    flow_problem.run(1,"linear",true);
    }
  catch (std::exception &exc)
    {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
    }
  catch (...)
    {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
    }

  return 0;
  }
