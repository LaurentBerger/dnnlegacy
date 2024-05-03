/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//                        (3-clause BSD License)
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// Copyright (C) 2024, Berger Laurent, all rights reserved.
//M*/
#include "../precomp.hpp"


#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <filesystem>

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "opencv-onnx.pb.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

opencv_onnx::ValueInfoProto fixedDynamicShape(opencv_onnx::ValueInfoProto &node, std::map<std::string, int>& valSubstitute)
{
    // node.type().tensor_type().shape().dim_size()
    // auto a = node.type(); TypeProto
    // auto b = node.type().tensor_type();TypeProto_Tensor
    // auto c = node.type().tensor_type().shape(); TensorShapeProto
    opencv_onnx::ValueInfoProto newNode;
    opencv_onnx::TypeProto* tp = new opencv_onnx::TypeProto();
    opencv_onnx::TypeProto_Tensor* tpt = new opencv_onnx::TypeProto_Tensor();
    opencv_onnx::TensorShapeProto* shapeNew = new opencv_onnx::TensorShapeProto();;
    shapeNew->Clear();
    shapeNew->clear_dim();

    newNode.CopyFrom(node);
    newNode.set_allocated_type(tp);
    tp->set_allocated_tensor_type(tpt);
    tpt->set_allocated_shape(shapeNew);
    for (int j = 0; j < node.type().tensor_type().shape().dim_size(); j++)
    {
        auto dim_src = node.type().tensor_type().shape().dim()[j];
        auto x = shapeNew->add_dim();

        if (dim_src.has_dim_param())
        {
            if (valSubstitute.find(dim_src.dim_param()) != valSubstitute.end())
            {
                std::string key(dim_src.dim_param());
                x->set_dim_value(valSubstitute[key]);
                std::cout << "change in input " << key << "\n";
            }
        }
        else
        {
            x->set_dim_value(dim_src.dim_value());
        }
    }
    return newNode;
}

namespace cv {
    namespace dnnlegacy {

        using namespace dnn;
        opencv_onnx::ModelProto model_proto_src, model_proto_dst;
        struct LayerInfo {
            int layerId;
            int outputId;
            int depth;
            LayerInfo(int _layerId = 0, int _outputId = 0, int _depth = CV_32F)
                :layerId(_layerId), outputId(_outputId), depth(_depth) {}
        };

        struct TensorInfo {
            int real_ndims;
            TensorInfo(int _real_ndims = 0) : real_ndims(_real_ndims) {}
        };

        cv::dnn::Net importOnnxWithFixedShape(const char* onnxFile, std::map<std::string, int>& valSubstitute)
        {
            CV_Assert(onnxFile);
            std::filesystem::path onnxPath(onnxFile);
            int tailleModele = std::filesystem::file_size(onnxPath);

            std::fstream input(onnxFile, std::ios::in | std::ios::binary);
            if (!input)
            {
                CV_Error(Error::StsBadArg, cv::format("Can't read ONNX file: %s", onnxFile));
            }

            if (!model_proto_src.ParseFromIstream(&input))
            {
                CV_Error(Error::StsUnsupportedFormat, cv::format("Failed to parse ONNX model: %s", onnxFile));
            }

            CV_Assert(model_proto_src.has_graph());
            opencv_onnx::GraphProto* graph_proto_src = model_proto_src.mutable_graph();
            opencv_onnx::GraphProto* graph_proto_dst = model_proto_dst.mutable_graph();
            std::string framework_version, framework_name;
            if (model_proto_src.has_producer_name())
            {
                framework_name = model_proto_src.producer_name();
                std::string* s = new std::string(framework_name);
                model_proto_dst.set_allocated_producer_name(s);
            }
            if (model_proto_src.has_producer_version())
            {
                framework_version = model_proto_src.producer_version();
                std::string* s = new std::string(framework_version);
                model_proto_dst.set_allocated_producer_version(s);
            }
            if (model_proto_src.has_doc_string())
                model_proto_dst.set_doc_string(model_proto_src.doc_string());
            if (model_proto_src.has_ir_version())
                model_proto_dst.set_ir_version(model_proto_src.ir_version());
            if (model_proto_src.has_domain())
            {
                model_proto_dst.set_domain(framework_version);
            }
            if (model_proto_src.has_model_version())
            {
                model_proto_dst.set_model_version(model_proto_src.model_version());
            }
            for (int i = 0; i < model_proto_src.metadata_props_size(); i++)
            {
                auto meta_src = model_proto_src.metadata_props(i);
                auto meta_dst = model_proto_dst.add_metadata_props();
                meta_dst->CopyFrom(meta_src);
            }
            for (int i = 0; i < model_proto_src.opset_import_size(); i++)
            {
                auto op_src = model_proto_src.opset_import(i);
                auto op_dst = model_proto_dst.add_opset_import();
                op_dst->CopyFrom(op_src);
            }

            // **************************************************************
            // ******************************************
            // ********************

            for (int i = 0; i < graph_proto_src->input_size(); i++)
            {
                auto node = graph_proto_src->input(i);
                auto newNode = fixedDynamicShape(node, valSubstitute);
                auto node_dst = graph_proto_dst->add_input();
                node_dst->CopyFrom(newNode);

            }
            // ********************
            // ******************************************
            // **************************************************************

            for (int i = 0; i < graph_proto_src->output_size(); i++)
            {
                auto node = graph_proto_src->output(i);
                auto newNode = fixedDynamicShape(node, valSubstitute);
                auto node_dst = graph_proto_dst->add_output();
                node_dst->CopyFrom(newNode);
            }
            for (int i = 0; i < graph_proto_src->node_size(); i++)
            {
                auto node = graph_proto_src->node(i);
                auto node_dst = graph_proto_dst->add_node();
                node_dst->CopyFrom(node);
            }
            for (int i = 0; i < graph_proto_src->initializer_size(); i++)
            {
                auto init = graph_proto_src->initializer(i);
                auto init_dst = graph_proto_dst->add_initializer();
                init_dst->CopyFrom(init);
            }
            for (int i = 0; i < graph_proto_src->value_info_size(); i++)
            {
                auto value = graph_proto_src->value_info(i);
                auto value_dst = graph_proto_dst->add_value_info();
                value_dst->CopyFrom(value);
            }
            std::ofstream fout("tmp.onnx", std::ios_base::binary);
            model_proto_dst.SerializePartialToOstream(&fout);
            fout.close();
            cv::dnn::Net n= cv::dnn::readNetFromONNX("tmp.onnx");
            return n;
            std::vector<uchar> tmpBuffer;
            tmpBuffer.resize(tailleModele + 1000);
            model_proto_dst.SerializeToArray(tmpBuffer.data(), tmpBuffer.size());
            return cv::dnn::readNetFromONNX(tmpBuffer);
        }
    }
}
