#ifndef _RegisterDWIToB0_CXX
#define _RegisterDWIToB0_CXX

#include "register_dwi_to_b0.h"
#include "rigid_register_images.h"
#include "registration_settings.h"

#include "itkMattesMutualInformationImageToImageMetricv4Okan.h"
#include "itkDIFFPREPGradientDescentOptimizerv4.h"
#include "itkOkanImageRegistrationMethodv4.h"
#include "TORTOISE.h"
#include "itkImageRegistrationMethodv4.h"
#include "../tools/ResampleDWIs/resample_dwis.h"



QuadraticTransformType::Pointer  RegisterDWIToB0(ImageType3D::Pointer fixed_img, ImageType3D::Pointer moving_img,std::string phase, MeccSettings *mecc_settings, bool initialize,std::vector<float> lim_arr, int vol,  QuadraticTransformType::Pointer minit_trans, bool retry_allowed)
{    
    int NITK= TORTOISE::GetAvailableITKThreadFor();

    typedef itk::MattesMutualInformationImageToImageMetricv4Okan<ImageType3D,ImageType3D> MetricType;
    MetricType::Pointer         metric        = MetricType::New();
    metric->SetNumberOfHistogramBins(mecc_settings->getNBins());
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetUseFixedImageGradientFilter(false);
    metric->SetFixedMin(lim_arr[0]);
    metric->SetFixedMax(lim_arr[1]);
    metric->SetMovingMin(lim_arr[2]);
    metric->SetMovingMax(lim_arr[3]);


    QuadraticTransformType::Pointer  initialTransform = QuadraticTransformType::New();
    initialTransform->SetPhase(phase);
    initialTransform->SetIdentity();
    if(minit_trans)
        initialTransform->SetParameters(minit_trans->GetParameters());


    QuadraticTransformType::ParametersType flags, grd_scales;
    flags.SetSize(QuadraticTransformType::NQUADPARAMS);
    flags.Fill(0);
    grd_scales.SetSize(QuadraticTransformType::NQUADPARAMS);
    for(int i=0;i<QuadraticTransformType::NQUADPARAMS;i++)
    {
        flags[i]= mecc_settings->getFlags()[i];
    }
    initialTransform->SetParametersForOptimizationFlags(flags);

    QuadraticTransformType::ParametersType init_params= initialTransform->GetParameters();
    if(!minit_trans)
    {
        init_params[0]=0;           //THIS IS NOW JUST DWI TO B=0 registration. No need to initialize. It is causing problems with spherical data
        init_params[1]=0;
        init_params[2]=0;
        init_params[21]=0;
        init_params[22]=0;
        init_params[23]=0;
    }

    initialTransform->SetParameters(init_params);


    ImageType3D::SizeType sz= fixed_img->GetLargestPossibleRegion().GetSize();
    ImageType3D::SpacingType res = fixed_img->GetSpacing();

    int ph=0;
    if(phase=="vertical")
        ph=1;
    if(phase=="slice")
        ph=2;

    if( mecc_settings->getGrdSteps().size()==24)
    {
        for(int i=0;i<24;i++)
            grd_scales[i]=  mecc_settings->getGrdSteps()[i];
    }
    else
    {
        grd_scales[0]= res[0]*1.25;
        grd_scales[1]= res[1]*1.25;
        grd_scales[2]= res[2]*1.25;

        grd_scales[3]=0.05;
        grd_scales[4]=0.05;
        grd_scales[5]=0.05;


        grd_scales[6]= res[2]*1.5 /   ( sz[0]/2.*res[0]    )*2;
        grd_scales[7]= res[2]*1.5 /   ( sz[1]/2.*res[1]    )*2.;
        grd_scales[8]= res[2]*1.5 /   ( sz[2]/2.*res[2]    )*2.;


        grd_scales[9]=  0.5*res[2]*10. /   ( sz[0]/2.*res[0]    ) / ( sz[1]/2.*res[1]    );
        grd_scales[10]= 0.5*res[2]*10. /   ( sz[0]/2.*res[0]    ) / ( sz[2]/2.*res[2]    );
        grd_scales[11]= 0.5*res[2]*10. /   ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    );

        grd_scales[12]= res[2]*5. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
        grd_scales[13]= res[2]*8. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    )/2.;

        grd_scales[14]= 2* 5.*res[2]*4 /   ( sz[0]/2.*res[0]    ) / ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    );

        grd_scales[15]=  2*5.*res[2]*1 /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
        grd_scales[16]= 2* 5.*res[2]*1. /   ( sz[1]/2.*res[1]    ) / ( sz[1]/2.*res[1]    ) / ( sz[1]/2.*res[1]    );
        grd_scales[17]= 2* 5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[2]/2.*res[2]    ) / ( sz[2]/2.*res[2]    );
        grd_scales[18]= 2* 5.*res[2]*1. /   ( sz[1]/2.*res[1]    ) / ( sz[2]/2.*res[2]    ) / ( sz[2]/2.*res[2]    );
        grd_scales[19]=  2*5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );
        grd_scales[20]= 2* 5.*res[2]*1. /   ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    ) / ( sz[0]/2.*res[0]    );

        grd_scales[21]= res[0]/1.25;
        grd_scales[22]= res[1]/1.25;
        grd_scales[23]= res[2]/1.25;
    }  

    if(initialize)
    {
        MetricType::Pointer         metric2        = MetricType::New();
        metric2->SetNumberOfHistogramBins((mecc_settings->getNBins()));
        metric2->SetUseMovingImageGradientFilter(false);
        metric2->SetUseFixedImageGradientFilter(false);

        typedef itk::DIFFPREPGradientDescentOptimizerv4<double> OptimizerType;
        OptimizerType::Pointer      optimizer     = OptimizerType::New();

        QuadraticTransformType::ParametersType flags2;
        flags2.SetSize(24);
        flags2.Fill(0);
        flags2[0]=mecc_settings->getFlags()[0];
        flags2[1]=mecc_settings->getFlags()[1];
        flags2[2]=mecc_settings->getFlags()[2];
        flags2[3]=mecc_settings->getFlags()[3];
        flags2[4]=mecc_settings->getFlags()[4];
        flags2[5]=mecc_settings->getFlags()[5];


        QuadraticTransformType::ParametersType grd_scales2= grd_scales;
        grd_scales2[3]=2*grd_scales2[3];
        grd_scales2[4]=2*grd_scales2[4];
        grd_scales2[5]=2*grd_scales2[5];

        optimizer->SetOptimizationFlags(flags2);
        optimizer->SetGradScales(grd_scales2);
        optimizer->SetNumberHalves(mecc_settings->getNumberHalves());
        optimizer->SetBrkEps(mecc_settings->getBrkEps());


        typedef itk::OkanImageRegistrationMethodv4<ImageType3D,ImageType3D, QuadraticTransformType,ImageType3D >           RegistrationType;
        RegistrationType::Pointer   registration  = RegistrationType::New();
        registration->SetFixedImage(fixed_img);
        registration->SetMovingImage(moving_img);
        registration->SetMetricSamplingPercentage(1.);        
        registration->SetOptimizer(optimizer);
        registration->SetInitialTransform(initialTransform);
        registration->InPlaceOn();        
        registration->SetNumberOfWorkUnits(NITK);
        //registration->SetNumberOfThreads(NITK);
        metric2->SetMaximumNumberOfWorkUnits(NITK);
        registration->SetMetric(        metric2        );



        RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
        shrinkFactorsPerLevel.SetSize( 1 );
        shrinkFactorsPerLevel[0] = 3;


        RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
        smoothingSigmasPerLevel.SetSize(1 );
        smoothingSigmasPerLevel[0] = 1.;


        registration->SetNumberOfLevels( 1 );
        registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
        registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
        registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
        try
          {
          registration->Update();
          }
        catch( itk::ExceptionObject & err )
          {
          std::cerr << "ExceptionObject caught inprerigid:!" << std::endl;
          std::cerr << err << std::endl;
          std::cerr<< "In volume: "<< vol<<std::endl;
          std::cerr<< initialTransform->GetParameters()<<std::endl;
          return nullptr;
          }        
    }    

    QuadraticTransformType::ParametersType finalParameters=initialTransform->GetParameters();    

        {
            using OptimizerType= itk::DIFFPREPGradientDescentOptimizerv4<double> ;
            OptimizerType::Pointer      optimizer     = OptimizerType::New();
            optimizer->SetOptimizationFlags(flags);
            optimizer->SetGradScales(grd_scales);
            optimizer->SetNumberHalves(mecc_settings->getNumberHalves());
            optimizer->SetBrkEps(mecc_settings->getBrkEps());



            typedef itk::OkanImageRegistrationMethodv4<ImageType3D,ImageType3D, QuadraticTransformType,ImageType3D >           RegistrationType;
            RegistrationType::Pointer   registration  = RegistrationType::New();
            registration->SetFixedImage(fixed_img);
            registration->SetMovingImage(moving_img);
            registration->SetMetricSamplingPercentage(1.);            
            registration->SetOptimizer(optimizer);
            registration->SetInitialTransform(initialTransform);
            registration->InPlaceOn();
            registration->SetNumberOfWorkUnits(NITK);
            //registration->SetNumberOfThreads(NITK);
            metric->SetMaximumNumberOfWorkUnits(NITK);
            registration->SetMetric(        metric        );


            RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
            shrinkFactorsPerLevel.SetSize( 3 );
            shrinkFactorsPerLevel[0] = 4;
            shrinkFactorsPerLevel[1] = 2;
            shrinkFactorsPerLevel[2] = 1;

            RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
            smoothingSigmasPerLevel.SetSize( 3 );
            smoothingSigmasPerLevel[0] = 1.;
            smoothingSigmasPerLevel[1] = 0.25;
            smoothingSigmasPerLevel[2] = 0.;

            registration->SetNumberOfLevels( 3 );
            registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
            registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);
            registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
            try
              {
              registration->Update();
              }
            catch( itk::ExceptionObject & err )
              {
              std::cerr << "ExceptionObject caught !" << std::endl;
              std::cerr << err << std::endl;
              std::cerr<< "In volume: "<< vol<<std::endl;
              std::cerr<< initialTransform->GetParameters()<<std::endl;
              return nullptr;
              }            
            finalParameters =  initialTransform->GetParameters();
        }



    // Quality check: if large_motion_correction is enabled, evaluate the metric
    // at the final transform. If it looks bad, fall back to multistart search.
    bool large_motion = RegistrationSettings::get().getValue<bool>(std::string("large_motion_correction"));
    if(large_motion && retry_allowed)
    {
        typedef itk::MattesMutualInformationImageToImageMetricv4Okan<ImageType3D,ImageType3D> QCMetricType;
        QCMetricType::Pointer qc_metric = QCMetricType::New();
        qc_metric->SetNumberOfHistogramBins(mecc_settings->getNBins());
        qc_metric->SetUseMovingImageGradientFilter(false);
        qc_metric->SetUseFixedImageGradientFilter(false);
        qc_metric->SetFixedMin(lim_arr[0]);
        qc_metric->SetFixedMax(lim_arr[1]);
        qc_metric->SetMovingMin(lim_arr[2]);
        qc_metric->SetMovingMax(lim_arr[3]);
        qc_metric->SetFixedImage(fixed_img);
        qc_metric->SetMovingImage(moving_img);
        qc_metric->SetMovingTransform(initialTransform);
        qc_metric->SetMaximumNumberOfWorkUnits(NITK);

        try
        {
            qc_metric->Initialize();
            double metric_value = qc_metric->GetValue();

            // Also compute identity metric for comparison
            QuadraticTransformType::Pointer id_trans = QuadraticTransformType::New();
            id_trans->SetPhase(phase);
            id_trans->SetIdentity();
            qc_metric->SetMovingTransform(id_trans);
            qc_metric->Initialize();
            double identity_metric = qc_metric->GetValue();

            // If registration made things worse than identity (metric is negative for MI,
            // more negative is better), fall back to multistart
            if(metric_value > identity_metric * 0.9)
            {
                // Registration likely failed -- try multistart
                std::vector<float> down_factors;
                std::vector<float> new_res(3);
                new_res[0] = fixed_img->GetSpacing()[0] * 2.0;
                new_res[1] = fixed_img->GetSpacing()[1] * 2.0;
                new_res[2] = fixed_img->GetSpacing()[2] * 2.0;
                ImageType3D::Pointer fixed_down  = resample_3D_image(fixed_img,  new_res, down_factors, "Linear");
                ImageType3D::Pointer moving_down = resample_3D_image(moving_img, new_res, down_factors, "Linear");

                RigidTransformType::Pointer ms_result = MultiStartRigidSearchCoarseToFine(fixed_down, moving_down, "MI");

                if(ms_result)
                {
                    // Convert to quadratic init and re-run
                    QuadraticTransformType::Pointer ms_init = QuadraticTransformType::New();
                    ms_init->SetPhase(phase);
                    ms_init->SetIdentity();
                    QuadraticTransformType::ParametersType ms_params = ms_init->GetParameters();
                    ms_params[0] = ms_result->GetOffset()[0];
                    ms_params[1] = ms_result->GetOffset()[1];
                    ms_params[2] = ms_result->GetOffset()[2];
                    ms_params[3] = ms_result->GetAngleX();
                    ms_params[4] = ms_result->GetAngleY();
                    ms_params[5] = ms_result->GetAngleZ();
                    ms_init->SetParameters(ms_params);

                    // Re-run the full registration with the multistart result as initialization
                    // Pass retry_allowed=false to prevent unbounded recursion
                    QuadraticTransformType::Pointer retry_result = RegisterDWIToB0(fixed_img, moving_img, phase, mecc_settings, initialize, lim_arr, vol, ms_init, false);
                    if(retry_result)
                        return retry_result;
                }
            }
        }
        catch(itk::ExceptionObject &e)
        {
            std::cerr << "QC metric evaluation failed: " << e.GetDescription() << std::endl;
        }
        catch(std::exception &e)
        {
            std::cerr << "QC metric evaluation failed: " << e.what() << std::endl;
        }
    }

    QuadraticTransformType::Pointer finalTransform = QuadraticTransformType::New();

    finalTransform->SetPhase(phase);
    finalTransform->SetIdentity();
    finalTransform->SetParametersForOptimizationFlags(flags);
    finalTransform->SetParameters( finalParameters );

    return finalTransform;
}




#endif
